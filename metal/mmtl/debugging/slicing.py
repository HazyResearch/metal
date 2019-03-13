import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

from metal.mmtl.debugging.lf_helpers import regex_present


def slice_dashsemicolon(row):
    if "-" in row["sentence1"] or ";" in row["sentence1"]:
        return True
    else:
        return False


def slice_endquestionword(row):
    word_list = row["sentence1"].split()
    if word_list[-2] in ["who", "what", "where", "when", "why", "how"]:
        return True
    else:
        return False


def slice_morepeople(row):
    people = 0
    sentence = row["sentence1"].split()
    for pronoun in ["she", "her", "hers"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["he", "him", "his"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["you", "your", "yours"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["I", "my", "me", "mine"]:
        if pronoun in sentence:
            people += 1
            break
    return people > 1


def slice_lowbleu(row):
    sentence1 = row["sentence1"].split()
    sentence2 = row["sentence2"].split()
    bleu = sentence_bleu([sentence1], sentence2)
    return bleu < 0.05
