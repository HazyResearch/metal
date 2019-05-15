import random
import re
from os import listdir
from os.path import isfile, join

MAX_FILES = 8000


def process_review(review):
    sentences = re.split("[.!:] ", review)
    filtered = []
    for sentence in sentences:
        n_words = len(sentence.strip().split(" "))
        if n_words < 2 or n_words > 25:
            continue
        filtered.append(sentence)
    return filtered


augmented_dir = "/dfs/scratch0/mccreery/GLUE_WS/"
output_fp = open(join(augmented_dir, "imdb_parsed.tsv"), "w+")

mypath = join(augmented_dir, "aclImdb/train/" + "pos")
posfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = join(augmented_dir, "aclImdb/train/" + "neg")
negfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

for i in range(MAX_FILES):
    input_fp = open(posfiles[i], "r")
    examples = process_review(input_fp.read())
    labels = ["1"] * len(examples)
    input_fp.close()

    input_fp = open(negfiles[i], "r")
    neg_examples = process_review(input_fp.read())
    examples = examples + neg_examples
    labels = labels + ["0"] * len(neg_examples)
    input_fp.close()

    shuffler = [i for i in range(len(examples))]
    random.shuffle(shuffler)
    for i in shuffler:
        output_fp.write(examples[i] + "\t" + labels[i] + "\n")

output_fp.close()
