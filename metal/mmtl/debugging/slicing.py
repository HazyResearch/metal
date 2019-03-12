import pandas as pd

from metal.mmtl.debugging.lf_helpers import regex_present


def slice_dashsemicolon(row):
    if "-" in row["sentence1"] or ";" in row["sentence1"]:
        return True
    else:
        return False


def slicing_function(row):
    people = 0
    for pronoun in ["she", "her", "hers"]:
        if regex_present(row, pronoun, fields=["sentence1"]):
            people += 1
            continue
    for pronoun in ["he", "him", "his"]:
        if regex_present(row, pronoun, fields=["sentence1"]):
            people += 1
            continue
    for pronoun in ["you", "your", "yours"]:
        if regex_present(row, pronoun, fields=["sentence1"]):
            people += 1
            continue
    for pronoun in ["I", "my", "me", "mine"]:
        if regex_present(row, pronoun, fields=["sentence1"]):
            people += 1
            continue
    return people > 3
