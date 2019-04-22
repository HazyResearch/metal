import argparse
import os

import dill

from metal.mmtl.glue.glue_datasets import get_glue_dataset


def make_datasets(task, bert_version):
    datasets = {}
    for split in ["train", "dev", "test"]:
        datasets[split] = get_glue_dataset(
            task, split, bert_version, max_len=200, run_spacy=True
        )
    return datasets


def pickle_datasets(datasets, task, bert_version):
    bert_str = bert_version.replace("-", "_")
    filename = f"{task}_{bert_str}_spacy_datasets"
    filepath = f"{os.environ['GLUEDATA']}/datasets/{filename}.dill"
    with open(filepath, "wb") as f:
        dill.dump(datasets, f)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    assert args.task.isupper()
    datasets = make_datasets(args.task, args.bert_version)

    if pickle_datasets(datasets, args.task, args.bert_version):
        print(f"FINISHED: {args.task}")
    else:
        print(f"FAILED: {args.task}")
