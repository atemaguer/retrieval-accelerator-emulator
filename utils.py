import pandas as pd


def get_dataset(dataset_path):
    df = pd.read_csv(dataset_path, sep="\t", names=["pid", "text"])
    dataset = [{"pid": x[0], "text": x[1]} for x in df.itertuples(index=False)]

    return dataset


def calculate_recall():
    pass
