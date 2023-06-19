import pandas as pd
from src.data import Dataset

dataset = Dataset("DB100K")

df = pd.read_csv("ranks.csv", sep=";")
df.drop("head_rank", axis=1, inplace=True)

df = df[df["tail_rank"] == 1]
df["head_id"] = df["head"].map(dataset.entity_to_id.get)
df["triples"] = df["head_id"].map(dataset.entity_to_training_triples.get)
df["triples_number"] = df["triples"].map(len)

df = df.sample(100)
df = df.reset_index(drop=True)
df.drop(["tail_rank", "head_id", "triples", "triples_number"], axis=1, inplace=True)

output_path = "preds/ConvE_DB100K.csv"
df.to_csv(output_path, sep="\t", index=False, header=False)
