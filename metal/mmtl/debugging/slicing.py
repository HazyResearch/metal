import pandas as pd


def slice_odd_punc(row):
    if "-" in row["sentence1"] or ";" in row["sentence1"] or ":" in row["sentence1"]:
        return True
    else:
        return False
