import os
from collections import defaultdict


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

    def get_examples(self, tag):
        """ Parses the uids for a particular tag and returns tuples of (uid, examples)
        from raw data e.g. ('RTE/dev.tsv:1, [..., sent1, setn2, label1, ...])

        NOTE: this can be improved with additional knowledge of where indexes are
        located.
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

            # take the raw line, remove \n, and split by \t for readability
            exs = [(uid, fn_lines[l].strip().split("\t")) for l in lines]
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
