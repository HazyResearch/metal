import pandas as pd
import spacy
from nltk.translate.bleu_score import sentence_bleu

nlp = spacy.load("en_core_web_sm")


def slice_who(row):
    sent2 = row["sentence2"].replace(" ##", "")
    sent2 = nlp(sent2)
    for ent in sent2.ents:
        if ent.label_ in ["PERSON"] and ent.text not in row["sentence1"]:
            return True  # event, person, org, gpe in question but not named in answer
    return False


def slice_entsecondonly(row):
    sent2 = nlp(row["sentence2"])
    for ent in sent2.ents:
        if ent.text not in row["sentence1"]:
            return True
    return False


def slice_function(row):
    sent1 = row["sentence1"].split()
    sent2 = row["sentence2"].split()
    word = "why"
    if word in sent1 and word not in sent2:
        return True
    if word in sent2 and word not in sent1:
        return True
    return False


def slice_numbers(row):
    sent1 = row["sentence1"]
    sent2 = row["sentence2"]
    for num in range(10):
        if str(num) in sent2 and str(num) not in sent1:
            return True
        if str(num) in sent1 and str(num) not in sent2:
            return True
    return False


def slice_firstsecondperson(row):
    sent1 = row["sentence1"].split()
    for word in ["you", "your", "yours", "we", "us", "our", "I", "my", "me"]:
        if word in sent1:
            return True
    return False


def slice_contraction(row):
    sent1 = row["sentence1"].split()
    if "'" in sent1:
        return True
    return False


def slice_qmark(row):
    sent1 = row["sentence1"].split()
    if sent1[-1] == "?":
        return True
    return False


def slice_qword(row):
    sent1 = row["sentence1"].split()
    for word in ["who", "what", "where", "when", "why", "how"]:
        if word in sent1 and "?" not in sent1:
            return True
    return False


def slice_quotes(row):
    sent1 = row["sentence1"].split()
    quotes = 0
    for tok in sent1:
        if tok == "'" or tok == '"':
            quotes += 1
    return quotes == 2


def slice_modals(row):
    sent1 = row["sentence1"].split()
    if any(
        [word in sent1 for word in ["could", "would", "should", "ought"]]
    ):  # could train on 'should' too
        return True
    return False


def slice_qualifier(row):
    sent1 = row["sentence1"].split()
    if any(
        [word in sent1 for word in ["while", "although", "aside", "but"]]
    ):  # could train on 'but' too
        return True
    return False


def slice_longsentence1(row, thresh):
    sent1 = row["sentence1"].split()
    if len(sent1) > thresh:
        return True
    return False


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
