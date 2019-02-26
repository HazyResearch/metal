import os


class Tagger(object):
    def __init__(
        self,
        tags_dir=os.path.join(os.environ["METALHOME"], "metal/mmtl/debugging/tags"),
    ):
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
            uids = map(lambda x: x + "\n", uids)
            f.writelines(uids)
            f.close()

    def get_uids(self, tag):
        tag_path = self._get_tag_path(tag)
        with open(tag_path, "r") as f:
            uids = [x.strip() for x in f.readlines()]
            return uids

    def remove_tag(self, uid, tag):
        tag_path = self._get_tag_path(tag)
        with open(tag_path, "r+") as f:
            uids = [x.strip() for x in open(tag_path, "r").readlines()]
            uids.remove(uid)

        with open(tag_path, "w+") as f:
            uids = ["%s\n" % uid for uid in uids]
            f.writelines(uids)
            f.close()
