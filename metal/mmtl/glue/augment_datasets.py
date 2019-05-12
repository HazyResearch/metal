import csv
import os
import random

import pandas as pd

from metal.mmtl.glue.glue_preprocess import (
    get_task_tsv_config,
    load_tsv,
    tsv_path_for_dataset,
)
from metal.mmtl.glue.word_categories import (
    comparatives,
    negations,
    possessive,
    qwords,
    temporal,
    wh_words,
)

##### USER SETTINGS ###########
source_dataset_name = "MRPC"
slice_type = "wh"
augmentation_type = "del"
save = False
MAX_LEN = 15
max_datapoints = 8000
split = "train"
###############################

if save:
    max_datapoints = 6000
else:
    max_datapoints = 1000

slice_mappings = {
    "wh": wh_words,
    "q": qwords,
    "neg": negations,
    "but": ["but"],
    "temp": temporal,
    "poss": possessive,
    "comp": comparatives,
}


def select_sentences(text):
    if slice_type == "questions":
        return get_questions(text)
    else:
        return get_sentences_with(text, slice_mappings[slice_type])


def augment_negative(example):
    if augmentation_type == "del":
        return remove_random_words(example)
    elif augmentation_type == "swap":
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


def save_to_cola(positive_examples, negative_examples):
    dest_dataset_name = "CoLA"
    dest_split = "_".join(
        split, "from" + source_dataset_name, slice_type, augmentation_type
    )
    dest_filename = tsv_path_for_dataset(dest_dataset_name, dest_split)
    with open(dest_filename, "w") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for example in positive_examples:
            writer.writerow([source_dataset_name, 1, "?", example])
        for example in negative_examples:
            writer.writerow([source_dataset_name, 0, "?", example])


def main():
    split = "train"
    filename = tsv_path_for_dataset(source_dataset_name, split)
    config = get_task_tsv_config(source_dataset_name.upper(), split)
    text_blocks, labels = load_tsv(
        filename,
        config["sent1_idx"],
        config["sent2_idx"],
        config["label_idx"],
        True,
        max_datapoints=max_datapoints,
    )
    positive_examples = []
    negative_examples = []
    for i in range(len(labels)):
        for text in text_blocks[i]:
            positive_example = select_sentences(text)
            if positive_example is not None:
                positive_examples.append(positive_example)
    for example in positive_examples:
        negative_example = augment_negative(example)
        if not save:
            print(example + " 1")
            print(negative_example + " 0")
        negative_examples.append(negative_example)
    if save:
        save_to_cola(positive_examples, negative_examples)


if __name__ == "__main__":
    main()
