import os


def tsv_path_for_dataset(dataset_name, dataset_split):
    return os.path.join(
        os.environ["GLUEDATA"], "{}/{}.tsv".format(dataset_name, dataset_split)
    )
