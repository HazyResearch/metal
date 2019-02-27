import re


def regex_present(row, rgx, fields=["sentence1", "sentence2"]):
    return any(re.search(rgx, row[field]) for field in fields)
