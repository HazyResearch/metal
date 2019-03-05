import os
from collections import defaultdict

import metal.mmtl.dataset as dataset_module


class Tagger(object):
    def __init__(
        self,
        tags_dir=os.path.join(os.environ["METALHOME"], "metal/mmtl/debugging/tags"),
    ):
        if not os.path.isdir(tags_dir):
            os.mkdir(tags_dir)
        self.tags_dir = tags_dir

    def _get_tag_path(self, tag):
        return os.path.join(self.tags_dir, f"{tag}.txt")

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
        print(f"Added 1 tag. Tag set '{tag}' contains {len(uids)} tags.")

    def get_uids(self, tag):
        tag_path = self._get_tag_path(tag)
        with open(tag_path, "r") as f:
            uids = [x.strip() for x in f.readlines()]
            return uids

    def get_examples(self, tag, bert_tokenizer_vocab=None):
        """ Parses the uids for a particular tag and return appropriate examples.

        NOTE: this is done with many assumptions about the naming of the UIDs
            e.g. "RTE/dev.txt:29 --> "{TASK}/{filename}:{line_number}

        Args:
            tag: name of tag for which we want to return examples
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
            fn_to_lines[filename].append(int(line_num))

        # to return: list of examples
        examples = []
        for fn, lines in fn_to_lines.items():
            path = os.path.join(os.environ["GLUEDATA"], fn)
            with open(path, "r") as f:
                fn_lines = f.readlines()

            task_name, file_suffix = fn.split("/")
            split = file_suffix.split(".tsv")[0]

            # initialize dataset to find sent/label indexes
            dataset_cls = getattr(dataset_module, task_name.upper() + "Dataset")
            dataset = dataset_cls(split, None, load_tsv=False)

            # take the raw line, remove \n, and split by \t for readability
            exs = []
            for line in lines:
                split_line = fn_lines[line].strip().split("\t")
                example = {
                    "sent1": split_line[dataset.sent1_idx],
                    "sent2": split_line[dataset.sent2_idx],
                    "label": split_line[dataset.label_idx],
                }
                uid = f"{fn}:{line}"
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
