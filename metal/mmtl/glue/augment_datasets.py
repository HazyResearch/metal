import os
import random

import pandas as pd

from metal.mmtl.glue.glue_preprocess import (
    get_task_tsv_config,
    load_tsv,
    tsv_path_for_dataset,
)


def get_questions(text):
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
    twopart = text.split("[. ]")
    for part in twopart:
        words = part.split(" ")
        num_words = len(words)
        if part[-1] == "?" and words[0].lower() in qwords and num_words < 15:
            return part


def get_sentences(text):
    print(text)


def get_sentences_containing(text, word_list):
    print(text)


def make_unacceptable(example):
    words = example.split(" ")
    num_words = len(words)
    idx = random.randint(0, num_words - 1)
    del words[idx]
    return " ".join(words)


def main():
    source_dataset_name = "QQP"
    split = "train"
    filename = tsv_path_for_dataset(source_dataset_name, split)
    config = get_task_tsv_config(source_dataset_name.upper(), split)
    text_blocks, labels = load_tsv(
        filename,
        config["sent1_idx"],
        config["sent2_idx"],
        config["label_idx"],
        True,
        max_datapoints=50,
    )
    positive_examples = []
    negative_examples = []
    for i in range(len(labels)):
        for text in text_blocks[i]:
            positive_example = get_questions(text)
            if positive_example is not None:
                positive_examples.append(positive_example)
    for example in positive_examples:
        negative_example = make_unacceptable(example)
        negative_examples.append(negative_example)

    dest_dataset_name = "CoLA"
    dest_split = split + "_" + dest_dataset_name + "from" + source_dataset_name
    dest_filename = tsv_path_for_dataset(dest_dataset_name, dest_split)
    config = get_task_tsv_config(dest_dataset_name.upper(), split)
    print(dest_filename)


if __name__ == "__main__":
    main()
