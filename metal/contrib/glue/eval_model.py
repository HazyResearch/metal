"""Script to compute predictions for GLUE test sets and create submission zip."""
import argparse
import datetime
import json
import os
import warnings
import zipfile
from collections import Counter

import numpy as np
import torch
from scipy.stats import mode

from metal.mmtl.glue.glue_preprocess import get_task_tsv_config
from metal.mmtl.glue.glue_tasks import (
    create_glue_dataloaders,
    create_glue_datasets,
    create_glue_tasks_payloads,
)
from metal.mmtl.metal_model import MetalModel, probs_to_preds
from metal.mmtl.payload import Payload

task_to_name_dict = {
    "QNLI": "QNLI",
    "STSB": "STS-B",
    "SST2": "SST-2",
    "COLA": "CoLA",
    "MNLI": {"mismatched": "MNLI-mm", "matched": "MNLI-m", "diagnostic": "AX"},
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


def save_tsv(predictions, task_name, state, submission_dir):
    if isinstance(task_to_name_dict[task_name], dict):
        file_name = task_to_name_dict[task_name][state]
    else:
        file_name = task_to_name_dict[task_name]
    file_path = os.path.join(submission_dir, f"{file_name}.tsv")
    with open(file_path, "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(predictions):
            if task_name != "STSB":
                f.write(f"{int(idx)}\t{str(pred)}\n")
            else:
                # STSB is a regression task so we directly return scores
                f.write(f"{int(idx)}\t" + "{:.3f}\n".format(pred))
    print("Saved TSV to: ", file_path)


def zipdir(submission_dir, zip_filename):
    # ziph is zipfile handle
    zipf = zipfile.ZipFile(
        os.path.join(submission_dir, zip_filename), "w", zipfile.ZIP_DEFLATED
    )
    for root, dirs, files in os.walk(submission_dir):
        for file in files:
            if file.split(".")[-1] == "tsv":
                filepth = os.path.join(root, file)
                zipf.write(filepth, os.path.basename(filepth))
    zipf.close()


def apply_inv_fn(preds, inv_label_fn):
    predicted_labels = [inv_label_fn(pred) for pred in preds.flatten()]
    return predicted_labels


def ensemble_preds_mv(predictions, inv_fns):
    """Majority vote for classification and average for regression."""
    ensemble_preds = {}
    for (task_name, state), Y_probs in predictions.items():
        Y_probs = np.array(Y_probs)
        if Y_probs.shape[0] > 1:
            warnings.warn(f"Ensembling {Y_probs.shape[0]} models for {task_name} task")
        if task_name != "STSB":
            scores = [probs_to_preds(y_probs) for y_probs in Y_probs]
            preds = np.array(
                [apply_inv_fn(score, inv_fns[task_name]) for score in scores]
            )
            # get majority vote for classification tasks
            final_pred = mode(preds, axis=0).mode
        else:
            preds = np.array(
                [apply_inv_fn(score, inv_fns[task_name]) for score in Y_probs]
            )
            # average scores for regression tasks
            final_pred = preds.mean(0)
        ensemble_preds[(task_name, state)] = list(final_pred.flatten())
    return ensemble_preds


def ensemble_preds_avg(predictions, inv_fns):
    """Average scores, then apply inv_label_fn."""
    ensemble_preds = {}
    for (task_name, state), Y_probs in predictions.items():
        preds = np.array(Y_probs)
        if preds.shape[0] > 1:
            warnings.warn(f"Ensembling {preds.shape[0]} models for {task_name} task")
        # average scores for regression tasks
        if task_name != "STSB":
            final_scores = np.array(
                [probs_to_preds(y_probs) for y_probs in preds.mean(0)]
            )
        else:
            final_scores = preds.mean(0)
        final_preds = apply_inv_fn(final_scores, inv_fns[task_name])
        ensemble_preds[(task_name, state)] = final_preds
    return ensemble_preds


def ensemble_preds_wavg(predictions, inv_fns):
    """Average scores with weights a/t model confidences, then apply inv_label_fn."""
    ensemble_preds = {}
    for (task_name, state), Y_probs in predictions.items():
        preds = np.array(Y_probs)
        if preds.shape[0] > 1:
            warnings.warn(f"Ensembling {preds.shape[0]} models for {task_name} task")
        weights = torch.softmax(torch.Tensor(preds), dim=0)
        if task_name == "STSB":
            final_scores = np.average(preds, axis=0, weights=weights)
        else:
            final_scores = np.array(
                [
                    probs_to_preds(y_probs)
                    for y_probs in np.average(preds, axis=0, weights=weights)
                ]
            )
        final_preds = apply_inv_fn(final_scores, inv_fns[task_name])
        ensemble_preds[(task_name, state)] = final_preds
    return ensemble_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create sumbission zip file for GLUE MTL challenge.", add_help=True
    )
    parser.add_argument(
        "--model_paths", type=str, help="Path to json dict with model paths."
    )
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
        "--zip_filename",
        type=str,
        help="Name for the submission zip.",
        default="submission.zip",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        help="Which data split to evaluate on ['test', 'dev']. "
        "If eval_split=test, this scrip will compute predictions on all test splits and save them into a zip. "
        "If eval_split=dev, the script will only compute and report metric on the dev set"
        "useful to debug before submitting to leaderboard.",
        default="test",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=-1,
        help="Maximum number of examples per datasets. For debugging purposes.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--ensemble_mode",
        type=str,
        default="mv",
        help="Mode to ensemble model predictions. majority vote or average scores: ['mv', 'avg']",
    )
    parser.add_argument(
        "--use_task_checkpoints",
        action="store_true",
        help="Whether to use global checkpoint or task checkpoints for MTL models.",
    )
    args = parser.parse_args()

    with open(args.model_paths) as f:
        models = json.load(f)
    if args.eval_split == "test":
        submission_dir = get_full_sumbisison_dir(args.submission_dir)
        os.mkdir(submission_dir)
        json.dump(models, open(os.path.join(submission_dir, "model_paths.json"), "w"))

    dl_kwargs = {"batch_size": args.batch_size, "shuffle": False}
    predictions = {}
    inv_fns = {}
    for names, model_dirs in models.items():
        task_names = names.split(",")

        for model_dir in model_dirs:

            with open(os.path.join(model_dir, "task_config.json")) as f:
                task_config = json.load(f)

            # TODO: find a nicer way to get task names
            # create model
            task_config["splits"] = []
            tasks, _ = create_glue_tasks_payloads(task_names, **task_config)
            model = MetalModel(tasks, verbose=False, device=args.device)
            if not args.use_task_checkpoints or len(tasks) == 1:
                # load model weights
                model_path = os.path.join(model_dir, "best_model.pth")
                print(f"Loading model checkpoint: {model_path}")
                model.load_weights(model_path)
                model.eval()

            for task in tasks:

                # only predict for specified tasks
                if task.name in task_names:

                    # reload task specific checkpoints for MTL models
                    if len(tasks) > 1 and args.use_task_checkpoints:
                        model_path = os.path.join(
                            model_dir, f"{task.name}_best_model.pth"
                        )
                        print(f"Loading task specific checkpoint: {model_path}")
                        model.load_weights(model_path)
                        model.eval()

                    # get dataloaders
                    if "MNLI" in task.name:
                        # need to predict on mismatched and matched
                        states = ["mismatched", "matched"]
                        splits = [f"{args.eval_split}_{state}" for state in states]
                        if args.eval_split == "test":
                            # predict on diagnostic dataset for submission
                            splits.append("diagnostic")
                            states.append("diagnostic")

                    else:
                        states = [None]
                        splits = [args.eval_split]

                    datasets = create_glue_datasets(
                        dataset_name=task.name,
                        splits=splits,
                        bert_vocab=task_config["bert_model"],
                        max_len=task_config["max_len"],
                        max_datapoints=args.max_datapoints,
                    )
                    data_loaders = create_glue_dataloaders(
                        datasets,
                        dl_kwargs=dl_kwargs,
                        split_prop=None,
                        splits=splits,
                        seed=123,
                    )

                    for i, (split, data_loader) in enumerate(data_loaders.items()):
                        state = states[i]
                        payload_name = f"{task.name}_{split}"
                        payload = Payload(payload_name, data_loader, [task.name], split)

                        Ys, Ys_probs, Ys_preds = model.predict_with_gold(
                            payload, [task.name], return_preds=True
                        )

                        if args.eval_split == "dev":
                            target_metrics = {task.name: None}
                            metrics_dict = {}
                            scorer = model.task_map[task.name].scorer
                            print(model_path)
                            task_metrics_dict = scorer.score(
                                Ys[task.name],
                                Ys_probs[task.name],
                                Ys_preds[task.name],
                                target_metrics=target_metrics[task.name],
                            )
                            print(task_metrics_dict)
                        else:
                            Y = np.array(Ys[task.name])
                            Y_probs = np.array(Ys_probs[task.name])

                            if (task.name, state) in predictions:
                                predictions[(task.name, state)].append(Y_probs)
                            else:
                                predictions[(task.name, state)] = [Y_probs]
                                inv_fns[task.name] = get_task_tsv_config(
                                    task.name, split
                                )["inv_label_fn"]

    if args.eval_split == "test":
        if len(predictions) != 11:
            # make sure all 11 files are present before zipping
            warnings.warn(
                f"You only specified model paths for {len(predictions)} tasks. "
                f"The submission zip will be malformed."
            )
        if args.ensemble_mode == "mv":
            ensemble_predictions = ensemble_preds_mv(predictions, inv_fns)
        elif args.ensemble_mode == "avg":
            ensemble_predictions = ensemble_preds_avg(predictions, inv_fns)
        elif args.ensemble_mode == "wavg":
            ensemble_predictions = ensemble_preds_wavg(predictions, inv_fns)
        else:
            raise Exception(f"Unrecognized ensemble mode: {args.ensemble_mode}")

        for (task_name, state), predicted_labels in ensemble_predictions.items():
            # save predictions on test set for submission
            save_tsv(predicted_labels, task_name, state, submission_dir)
            if args.copy_models:
                # copy model weights and config to submission dir
                model_dir = os.path.dirname(model_path)
                os.system(f"cp -r {model_dir} {submission_dir}")

        zipdir(submission_dir, args.zip_filename)
