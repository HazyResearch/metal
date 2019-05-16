import argparse
import csv
import os
import random

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

from metal.mmtl.glue.glue_preprocess import (
    get_task_tsv_config,
    load_tsv,
    tsv_path_for_dataset,
)
from metal.mmtl.glue.word_categories import (
    common,
    comparatives,
    negations,
    possessive,
    qwords,
    temporal,
    wh_words,
)

nltk.download("wordnet")


slice_type = "wh"
MAX_LEN = 15

slice_mappings = {
    "wh": wh_words,
    "q": qwords,
    "neg": negations,
    "but": ["but"],
    "temp": temporal,
    "poss": possessive,
    "comp": comparatives,
}


def select_sentences(text, slice_type):
    if slice_type == "questions":
        return get_questions(text)
    elif slice_type == "all":
        num_words = len(text.split(" "))
        if num_words > 1 and num_words < MAX_LEN:
            return text
        else:
            return None
    else:
        return get_sentences_with(text, slice_mappings[slice_type])


def replace_synonyms(example):
    words = example.split(" ")
    num_words = len(words)
    idx1 = random.randint(0, num_words - 1)
    synonyms = []
    max_tries = 5
    while (words[idx1] in common or len(synonyms) < 2) and (max_tries > 0):
        idx1 = random.randint(0, num_words - 1)
        synonyms = []
        for syn in wn.synsets(words[idx1]):
            for l in syn.lemmas():
                synonyms.append(l.name())
        max_tries -= 1
    print(example)
    print(words[idx1])
    print(set(synonyms))
    print()


def augment_negative(example):
    words = example.split(" ")
    if len(words) <= 3:
        return remove_random_words(example)
    coin = random.randint(0, 1)
    if coin == 0:
        return remove_random_words(example)
    else:
        return swap_random_words(example)


def get_questions(text):
    twopart = preprocess_text(text)
    for part in twopart:
        part = fix_capitalization(part)
        words = part.split(" ")
        num_words = len(words)
        if num_words > 0 and num_words < MAX_LEN:
            if part[-1] == "?" and words[0].lower() in qwords:
                return part


def get_sentences_with(text, searchwords):
    twopart = preprocess_text(text)
    for part in twopart:
        part = fix_capitalization(part)
        words = part.split(" ")
        num_words = len(words)
        if num_words > 1 and num_words < MAX_LEN:
            for searchword in searchwords:
                if part[-1] == "." and searchword in words:
                    return part
                elif part[-1] != "?" and searchword in words:
                    return part + "."


def get_comparative_sentences(text):
    twopart = preprocess_text(text)
    for part in twopart:
        part = fix_capitalization(part)
        words = part.split(" ")
        num_words = len(words)
        if num_words > 1 and num_words < MAX_LEN:
            for word in words:
                if word in comparatives:
                    return part
                elif len(word) >= 3 and (word[-3:] in ["est", "ier"]):
                    # many false positives!
                    return part


def preprocess_text(text):
    text = text.replace('"', "")
    text = text.replace(".'", ".")
    twopart = text.split(". |.\t|.\n")
    return twopart


def fix_capitalization(sentence):
    sentence = sentence.strip()
    sentence = sentence[0].upper() + sentence[1:]
    sentence = sentence.replace(" i ", " I ")
    sentence = sentence.replace(" i'", " I'")
    return sentence


def remove_random_words(example):
    # TODO: detect adjectives and check against removing those
    words = example.split(" ")
    num_words = len(words)
    idx1 = random.randint(0, num_words - 1)
    del words[idx1]
    if num_words > 3:
        idx2 = random.randint(0, num_words - 2)
        del words[idx2]
    if (0 == idx1) or (num_words > 3 and 0 == idx2):
        words[0] = words[0].capitalize()
    if (num_words - 1 == idx1) or (num_words > 3 and (num_words - 2) == idx2):
        return " ".join(words) + "."
    else:
        return " ".join(words)


def swap_random_words(example):
    words = example.split(" ")
    num_words = len(words)
    idx1 = 0
    idx2 = 0
    while idx1 == idx2:
        idx1 = random.randint(1, num_words - 2)
        idx2 = random.randint(1, num_words - 2)
    temp = words[idx1]
    words[idx1] = words[idx2]
    words[idx2] = temp
    return " ".join(words)


def save_to_cola(all_examples, all_labels, args):
    # shuffler = [i for i in range(len(all_labels))]
    # random.shuffle(shuffler)
    dest_dataset_name = "CoLA"
    dest_split = "_".join([args.datasplit, "from" + args.sourcetask, args.slicetype])
    dest_filename = tsv_path_for_dataset(dest_dataset_name, dest_split)
    with open(dest_filename, "w") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for i in range(len(all_examples)):
            writer.writerow([args.sourcetask, all_labels[i], "?", all_examples[i]])


def get_commandline_args():
    parser = argparse.ArgumentParser(
        description="Augment data from one task for use with another (default CoLA).",
        add_help=False,
    )
    parser.add_argument(
        "--slicetype",
        type=str,
        default="wh",
        help="slice of data you want. Choose between: wh, q, neg, but, temp, poss, comp",
    )
    parser.add_argument("--datasplit", type=str, default="train", help="train/dev/test")
    parser.add_argument("--sourcetask", type=str, default="MNLI", help="Source Task")
    parser.add_argument("--desttask", type=str, default="MNLI", help="Source Task")
    parser.add_argument(
        "--save",
        action="store_true",
        help="If flag is used, behavior is to save output, otherwise to print to stdout",
    )
    return parser.parse_args()


fix = {
    "CoLA": "COLA",
    "STS-B": "STSB",
    "SST-2": "SST2",
    "QQP": "QQP",
    "MNLI": "MNLI",
    "QNLI": "QNLI",
    "WNLI": "WNLI",
    "MRPC": "MRPC",
    "RTE": "RTE",
}


def main():
    args = get_commandline_args()
    filename = tsv_path_for_dataset(args.sourcetask, args.datasplit)
    config = get_task_tsv_config(fix[args.sourcetask], args.datasplit)
    text_blocks, og_labels = load_tsv(
        filename,
        config["sent1_idx"],
        config["sent2_idx"],
        config["label_idx"],
        True,
        max_datapoints=10000 if args.save else 1000,
    )
    examples = []
    labels = []
    for i in range(len(text_blocks)):
        for text in text_blocks[i]:
            prob_neg = random.randint(1, 5)
            if args.sourcetask == args.desttask:  # can't assume well-formed
                if og_labels[i] == "1":
                    examples.append(text)
                    labels.append(1)
                    replace_synonyms(text)
                else:
                    examples.append(text)
                    labels.append(0)
            else:  # assume well-formed until we augment if from another task
                positive_example = select_sentences(text, args.slicetype)
                if positive_example is not None:
                    examples.append(positive_example)
                    labels.append(1)
                    if prob_neg < 4:  # match split of original dataset
                        negative_example = augment_negative(positive_example)
                        examples.append(negative_example)
                        labels.append(0)
        if len(examples) > 30000:
            break
    if args.save:
        save_to_cola(examples, labels, args)
    else:
        for i in range(len(examples)):
            print("{} {}".format(examples[i], labels[i]))


if __name__ == "__main__":
    main()
