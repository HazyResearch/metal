"""Script to compute predictions for GLUE test sets and create submission zip."""
import argparse
import datetime
import json
import os
import zipfile

import numpy as np

from metal.mmtl.glue_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.utils.dataset_utils import get_all_dataloaders

task_to_name_dict = {
    "QNLI": "QNLI",
    "STSB": "STS-B",
    "SST2": "SST-2",
    "COLA": "CoLA",
    "MNLI": {"mismatched": "MNLI-mm", "matched": "MNLI-m", "diagnostic": "diagnostic"},
    "MNLI_SAN": {
        "mismatched": "MNLI-mm",
        "matched": "MNLI-m",
        "diagnostic": "diagnostic",
    },
    "RTE": "RTE",
    "RTE_SAN": "RTE",
    "WNLI": "WNLI",
    "WNLI_SAN": "WNLI",
    "QQP": "QQP",
    "QQP_SAN": "QQP",
    "MRPC": "MRPC",
    "MRPC_SAN": "MRPC",
    "AX": "AX",
}


def get_full_sumbisison_dir(submission_dir):
    d = datetime.datetime.today()
    submission_dir = os.path.join(submission_dir, f"{d.day}_{d.month}_{d.year}")
    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)
    existing_dirs = np.array(
        [
            d
            for d in os.listdir(submission_dir)
            if os.path.isdir(os.path.join(submission_dir, d))
        ]
    ).astype(np.int)
    if len(existing_dirs) > 0:
        submission_count = str(existing_dirs.max() + 1)
    else:
        submission_count = "0"
    return os.path.join(submission_dir, submission_count)


def get_model_paths(model_paths_list):
    """Parse model path list and verify that 9 checkpoints are provided, with exactly only per task."""
    task_count = 0
    model_paths = []
    seen_tasks = []
    for model_path in model_paths_list:
        task_names = []
        for task_name in task_to_name_dict:
            if task_name in model_path:
                assert task_name not in seen_tasks
                task_names.append(task_name)
                seen_tasks.append(task_name)
                task_count += 1
        # model_paths_dict[','.join(task_names)] = model_path
        model_paths.append((task_names, model_path))
    assert task_count == 9
    return model_paths


def save_tsv(predictions, task_name, state, submission_dir):
    if isinstance(task_to_name_dict[task_name], dict):
        file_name = task_to_name_dict[task_name][state]
    else:
        file_name = task_to_name_dict[task_name]
    file_path = os.path.join(submission_dir, f"{file_name}.tsv")
    with open(file_path, "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(predictions):
            f.write(f"{int(idx)}\t{str(pred)}\n")
    print("Saved TSV to: ", file_path)


def zipdir(submission_dir):
    # ziph is zipfile handle
    zipf = zipfile.ZipFile(
        os.path.join(submission_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED
    )
    for root, dirs, files in os.walk(submission_dir):
        for file in files:
            if file.split(".")[-1] == "tsv":
                filepth = os.path.join(root, file)
                zipf.write(filepth, os.path.basename(filepth))
    zipf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create sumbission zip file for GLUE MTL challenge.", add_help=True
    )
    parser.add_argument("--model_paths", action="append", type=str)
    parser.add_argument("--device", type=int, help="0 for gpu, -1 for cpu", default=0)
    parser.add_argument(
        "--copy_models",
        action="store_true",
        help="Whether to copy models to submission dir",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        help="Where to save submission zip.",
        default=os.path.join(os.environ["METALHOME"], "submissions"),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        help="Which data split to evaluate on ['test', 'dev']."
        "If eval_split=test, this scrip will compute predictions on all test splits and save them into a zip."
        "If eval_split=dev, the script will only compute and report metric on the dev set"
        "useful to debug before submitting to leaderboard.",
        default="test",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Which bert model to use.",
    )
    parser.add_argument(
        "--bert_output_dim", type=int, default=768, help="Bert model output dimension."
    )
    parser.add_argument(
        "--max_len", type=int, default=256, help="Maximum sequence length."
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=-1,
        help="Maximum number of examples per datasets. For debugging purposes.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    args = parser.parse_args()

    if args.eval_split == "test":
        submission_dir = get_full_sumbisison_dir(args.submission_dir)
        os.mkdir(submission_dir)
        json.dump(
            args.model_paths,
            open(os.path.join(submission_dir, "model_paths.json"), "w"),
        )

    model_paths = get_model_paths(args.model_paths)
    all_scores = []

    dl_kwargs = {"batch_size": args.batch_size, "shuffle": False}

    for task_names, model_path in model_paths:

        # create model
        tasks = create_tasks(
            task_names=task_names,
            bert_model=args.bert_model,
            max_len=args.max_len,
            dl_kwargs=dl_kwargs,
            bert_output_dim=args.bert_output_dim,
            splits=[args.eval_split],
            max_datapoints=args.max_datapoints,
        )

        # load model weights
        model = MetalModel(tasks, verbose=False, device=args.device)
        model.load_weights(model_path)
        model.eval()
        for task in tasks:

            if "MNLI" in task.name:
                # need to predict on mismatched and matched
                states = ["mismatched", "matched"]
                splits = [f"{args.eval_split}_{state}" for state in states]
                if args.eval_split == "test":
                    # predict on diagnostic dataset for submission
                    splits.append("diagnostic")
                    states.append("diagnostic")
                task.data_loaders = get_all_dataloaders(
                    task.name,
                    args.bert_model,
                    max_len=args.max_len,
                    dl_kwargs=dl_kwargs,
                    max_datapoints=args.max_datapoints,
                    splits=splits,
                    split_prop=None,
                )
            else:
                states = [None]
                splits = [args.eval_split]

            if args.eval_split == "dev":
                # just compute evaluation metrics for debugging
                score = task.scorer.score(model, task)
                all_scores.append(score)

            else:
                for i, split in enumerate(splits):
                    # predict on test set
                    Y, Y_probs, Y_preds = model._predict_probs(
                        task, split=split, return_preds=True
                    )

                    inv_label_fn = task.data_loaders[split].dataset.inv_label_fn

                    if task.name != "STSB":
                        predicted_labels = [inv_label_fn(pred) for pred in Y_preds]
                    else:
                        # STSB is a regression task so we directly return scores
                        predicted_labels = [
                            round(inv_label_fn(pred), 3) for pred in Y_probs.flatten()
                        ]

                    # save predictions on test set for submission
                    save_tsv(predicted_labels, task.name, states[i], submission_dir)

                # copy model weights and config to submission dir
                model_dir = os.path.dirname(model_path)

        if args.eval_split == "test" and args.copy_models:
            os.system(f"cp -r {model_dir} {submission_dir}")

    if args.eval_split == "test":
        zipdir(submission_dir)
    else:
        for score in all_scores:
            print(score)
