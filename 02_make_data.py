import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ds = load_dataset("fancyzhx/ag_news", split="train")
df = ds.to_pandas()[["text", "label"]]
df = df.rename(columns={"label": "target"})
df = df.sample(20000, random_state=0).reset_index(drop=True)

print("类别分布:")
print(df["target"].value_counts().sort_index())
print("\n前 2 条:")
print(df.head(2).to_string())

train, val = train_test_split(df, test_size=0.15, random_state=0, stratify=df["target"])
train.to_csv("data/split/train.csv", index=False)
val.to_csv("data/split/val.csv", index=False)
print(f"\ntrain: {len(train)}  val: {len(val)}") 