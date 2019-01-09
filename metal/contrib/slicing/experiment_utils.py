import numpy as np
from termcolor import colored

from metal.metrics import metric_score


def compare_LF_slices(
    Yp_ours, Yp_base, Y, L_test, LFs, metric="accuracy", delta_threshold=0
):
    """Compares improvements between `ours` over `base` predictions."""

    improved = 0
    for LF_num, LF in enumerate(LFs):
        LF_covered_idx = np.where(L_test[:, LF_num] != 0)[0]
        ours_score = metric_score(
            Y[LF_covered_idx], Yp_ours[LF_covered_idx], metric
        )
        base_score = metric_score(
            Y[LF_covered_idx], Yp_base[LF_covered_idx], metric
        )

        delta = ours_score - base_score
        # filter out trivial differences
        if abs(delta) < delta_threshold:
            continue

        to_print = (
            f"[{LF.__name__}] delta: {delta:.4f}, "
            f"OURS: {ours_score:.4f}, BASE: {base_score:.4f}"
        )

        if ours_score > base_score:
            improved += 1
            print(colored(to_print, "green"))
        elif ours_score < base_score:
            print(colored(to_print, "red"))
        else:
            print(to_print)

    print(f"improved {improved}/{len(LFs)}")
