import csv
import os
import random

import pandas as pd

from metal.mmtl.glue.glue_preprocess import (
    get_task_tsv_config,
    load_tsv,
    tsv_path_for_dataset,
)

MAX_LEN = 15
temporal = ["before", "after", "past"]
spatial = ["inside", "within", "with"]
comparatives = [
    "more",
    "less",
    "bigger",
    "smaller",
    "better",
    "worse",
    "shorter",
    "taller",
    "most",
    "least",
]
negations = [
    "no",
    "not",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "never",
    "hardly",
    "scarcely",
    "barely",
    "doesn’t",
    "isn’t",
    "wasn’t",
    "shouldn’t",
    "wouldn’t",
    "couldn’t",
    "won’t",
    "can’t",
    "don’t",
]
qwords = [
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "will",
    "do",
    "does",
    "if",
    "which",
    "should",
    "could",
    "can",
]
wh_words = ["who", "what", "where", "when", "why", "which", "how"]


def get_questions(text):
    text = text.replace('"', "")
    text = text.replace(".'", ".")
    twopart = text.split(". |.\t|.\n")
    for part in twopart:
        part = fix_capitalization(part)
        words = part.split(" ")
        num_words = len(words)
        if num_words > 0 and num_words < MAX_LEN:
            if part[-1] == "?" and words[0].lower() in qwords:
                return part


def get_sentences_with(text, searchwords):
    text = text.replace('"', "")
    text = text.replace(".'", ".")
    twopart = text.split(". |.\t|.\n")
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


def fix_capitalization(sentence):
    sentence = sentence.strip()
    sentence = sentence[0].upper() + sentence[1:]
    sentence = sentence.replace(" i ", " I ")
    return sentence


def remove_random_words(example):
    # TODO: detect adjectives and check against removing those
    words = example.split(" ")
    num_words = len(words)
    idx1 = random.randint(0, num_words - 1)
    idx2 = random.randint(0, num_words - 2)
    del words[idx1]
    del words[idx2]
    if 0 in [idx1, idx2]:
        words[0] = words[0].capitalize()
    if ((num_words - 1) == idx1) or ((num_words - 2) == idx2):
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


def main():
    save = False
    source_dataset_name = "MNLI"
    split = "train"
    filename = tsv_path_for_dataset(source_dataset_name, split)
    config = get_task_tsv_config(source_dataset_name.upper(), split)
    text_blocks, labels = load_tsv(
        filename,
        config["sent1_idx"],
        config["sent2_idx"],
        config["label_idx"],
        True,
        max_datapoints=5000,
    )
    positive_examples = []
    negative_examples = []
    for i in range(len(labels)):
        for text in text_blocks[i]:
            positive_example = get_sentences_with(text, negations)
            if positive_example is not None:
                positive_examples.append(positive_example)
    for example in positive_examples:
        negative_example = remove_random_words(example)
        if not save:
            print(example + " 1")
            print(negative_example + " 0")
        negative_examples.append(negative_example)

    dest_dataset_name = "CoLA"
    dest_split = split + "_" + dest_dataset_name + "from" + source_dataset_name
    dest_filename = tsv_path_for_dataset(dest_dataset_name, dest_split)
    config = get_task_tsv_config(dest_dataset_name.upper(), split)
    if save:
        with open(dest_filename, "w") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            for example in positive_examples:
                writer.writerow([source_dataset_name, 1, "?", example])
            for example in negative_examples:
                writer.writerow([source_dataset_name, 0, "?", example])


if __name__ == "__main__":
    main()
