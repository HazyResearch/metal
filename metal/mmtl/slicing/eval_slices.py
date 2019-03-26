"""
Example script:
python eval_slices.py --tasks COLA --model_dir /dfs/scratch0/vschen/metal-mmtl/metal/mmtl/aws/output/2019_03_06_03_26_06/0/logdir/search_large_lr/QNLI.STSB.MRPC.QQP.WNLI.RTE.MNLI.SST2.COLA_11_43_53 --slices locs_orgs,proper_nouns
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from metal.mmtl.debugging.tagger import Tagger
from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads
from metal.mmtl.metal_model import MetalModel


def get_task_config(model_dir):
    with open(os.path.join(model_dir, "task_config.json")) as f:
        task_config = json.load(f)
    return task_config


def get_slice_metrics(task, Y, Y_probs, Y_preds, mask=None):
    if mask is None:
        mask = np.ones(len(Y)).astype(bool)

    return {
        "score": task.scorer.score(Y[mask], Y_probs[mask], Y_preds[mask]),
        "num_examples": np.sum(mask),
    }


def eval_on_slices(model_dir, task_names, slice_names, split="dev"):

    # initialize tasks/payloads with previous task_config
    task_config = get_task_config(model_dir)
    bert_model = task_config["bert_model"]
    max_len = task_config["max_len"]

    dl_kwargs = {"shuffle": False}
    tasks, payloads = create_glue_tasks_payloads(
        task_names=task_names,
        bert_model=bert_model,
        max_len=max_len,
        dl_kwargs=dl_kwargs,
        splits=[split],
        max_datapoints=-1,
        generate_uids=True,  # NOTE: this must be True to match with slice_uids!
    )

    # initialize model with previous weights
    pickled_model = os.path.join(model_dir, "model.pkl")
    try:
        model = torch.load(pickled_model)

    # model.pkl not found, or original file structure changed
    except (FileNotFoundError, ModuleNotFoundError):
        print(f"Unable to load {pickled_model}... loading weights instead.")
        # load weights instead
        model = MetalModel(tasks, verbose=False, device=0)
        model_path = os.path.join(model_dir, "best_model.pth")
        model.load_weights(model_path)

    # match uids for slices and evaluate
    tagger = Tagger(verbose=False)
    slice_scores = defaultdict(dict)
    for task, payload in zip(tasks, payloads):
        payload_uids = payload.data_loader.dataset.uids
        Ys, Ys_probs, Ys_preds = model.predict_with_gold(
            payload, [task.name], return_preds=True
        )
        Y = np.array(Ys[task.name])
        Y_probs = np.array(Ys_probs[task.name])
        Y_preds = np.array(Ys_preds[task.name])
        assert len(payload_uids) == len(Y) == len(Y_probs)

        # compute overall scores for task
        slice_scores[task.name].update(
            {"overall": get_slice_metrics(task, Y, Y_probs, Y_preds)}
        )

        # compute slice-specific scores
        for slice_name in slices_to_evaluate:
            # mask uids in slice
            slice_uids = tagger.get_uids(slice_name)
            mask = [uid in slice_uids for uid in payload_uids]
            mask = np.array(mask, dtype=bool)
            if np.sum(mask) == 0:
                print(f"No examples found... skipping {slice_name}")
                continue

            print(
                f"Found {np.sum(mask)}/{len(slice_uids)} "
                f"{slice_name} uids in {payload.name}"
            )

            slice_scores[task.name].update(
                {slice_name: get_slice_metrics(task, Y, Y_probs, Y_preds, mask)}
            )

    return dict(slice_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", required=True, type=str, help="Comma-sep task list e.g. QNLI,QQP"
    )
    parser.add_argument(
        "--slices", required=True, type=str, help="Comma-sep list of slices to evaluate"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="directory where *_config.json and <model>.pth are stored",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help="split to evaluate",
    )
    args = parser.parse_args()

    task_names = [task_name for task_name in args.tasks.split(",")]
    slices_to_evaluate = [slice_name for slice_name in args.slices.split(",")]
    slice_scores = eval_on_slices(
        args.model_dir, task_names, slices_to_evaluate, split=args.split
    )
    print(slice_scores)
