import os
from collections import defaultdict

from pytorch_pretrained_bert import BertTokenizer

from metal.mmtl.glue.glue_preprocess import get_task_tsv_config, load_tsv


class Tagger(object):
    def __init__(self, tags_dir="tags", verbose=True):
        parent_dir = os.path.join(os.environ["METALHOME"], "metal/mmtl/debugging/")
        tags_dir = os.path.join(parent_dir, tags_dir)
        if not os.path.isdir(tags_dir):
            os.mkdir(tags_dir)
        self.tags_dir = tags_dir
        self.verbose = verbose

    def _get_tag_path(self, tag):
        if "." not in tag:
            return os.path.join(self.tags_dir, f"{tag}.txt")
        else:
            return os.path.join(self.tags_dir, f"{tag}")

    def clear_tag(self, tag):
        tag_path = self._get_tag_path(tag)
        os.remove(tag_path)

    def add_tag(self, uid, tag):
        tag_path = self._get_tag_path(tag)

        if not os.path.exists(tag_path):
            print(f"Creating {tag_path}")
            with open(tag_path, "w"):
                pass

        with open(tag_path, "r+") as f:
            uids = set([x.strip() for x in open(tag_path, "r").readlines()])
            uids.add(uid)
            uids = sorted(map(lambda x: x + "\n", uids))
            f.writelines(uids)
            f.close()
        if self.verbose:
            print(f"Added 1 tag. Tag set '{tag}' contains {len(uids)} tags.")

    def get_uids(self, tag):
        tag_path = self._get_tag_path(tag)
        with open(tag_path, "r") as f:
            uids = [x.strip() for x in f.readlines()]
            return uids

    def get_examples(self, tag, bert_vocab=None):
        """ Parses the uids for a particular tag and return appropriate examples.
        NOTE: this is done with many assumptions about the naming of the UIDs
            e.g. "RTE/dev.txt:29 --> "{TASK}/{filename}:{line_number-1}

        Args:
            tag: name of tag for which we want to return examples
            bert_vocab: vocab file for bert tokenizer
        Returns:
            tuples of (uid, examples) from raw data
                e.g. ('RTE/dev.tsv:1, [..., sent1, sent2, label1, ...])

        """
        assert "GLUEDATA" in os.environ

        uids = self.get_uids(tag)
        # map filenames to line numbers for each
        fn_to_lines = defaultdict(list)
        for uid in uids:
            filename, line_num = uid.split(":")
            fn_to_lines[filename].append(int(line_num) - 1)

        # to return: list of examples
        examples = []
        for fn, lines in fn_to_lines.items():
            path = os.path.join(os.environ["GLUEDATA"], fn)
            with open(path, "r") as f:
                fn_lines = f.readlines()

            task_name, file_suffix = fn.split("/")
            task_name = task_name.replace("-", "")
            split = file_suffix.split(".tsv")[0]

            tokenizer = None
            if bert_vocab:
                do_lower_case = "uncased" in bert_vocab
                tokenizer = BertTokenizer.from_pretrained(
                    bert_vocab, do_lower_case=do_lower_case
                )

            # take the raw line, remove \n, and split by \t for readability
            exs = []
            config = get_task_tsv_config(task_name.upper(), split)
            for line in lines:
                uid = f"{fn}:{line+1}"
                try:
                    split_line = fn_lines[line].strip().split("\t")
                except IndexError:
                    print("Error:", line, uid)
                    continue

                sent1 = (
                    split_line[config["sent1_idx"]]
                    if config["sent1_idx"] >= 0
                    else None
                )
                sent2 = (
                    split_line[config["sent2_idx"]]
                    if config["sent2_idx"] >= 0
                    else None
                )
                example = {
                    "sent1": tokenizer.tokenize(sent1) if tokenizer else sent1,
                    "sent2": tokenizer.tokenize(sent2) if tokenizer else sent2,
                    "label": split_line[config["label_idx"]]
                    if config["label_idx"] >= 0
                    else None,
                }
                exs.append((uid, example))

            examples.extend(exs)
        return examples

    def remove_tag(self, uid, tag):
        tag_path = self._get_tag_path(tag)
        with open(tag_path, "r+") as f:
            uids = [x.strip() for x in open(tag_path, "r").readlines()]
            uids.remove(uid)

        with open(tag_path, "w+") as f:
            uids = ["%s\n" % uid for uid in uids]
            f.writelines(uids)
            f.close()
