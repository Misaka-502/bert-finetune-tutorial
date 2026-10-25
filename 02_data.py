# 02_data.py —— 第 2 课：把 csv 变成 (B, L) 矩阵
#
# 前置：已跑过 02_make_data.py，data/split/train.csv 存在
# 运行：.\run 02_data.py
#
# 本文件做三件事：
#   1) TextDataset  —— 按下标取一条样本（tokenize 在这里，不做 padding）
#   2) collate_fn   —— 把 N 条不等长样本拼成一个矩阵（padding 只在这里）
#   3) main()       —— 自测，验证"动态 padding"真的在发生
#
# 验收标准（运行后逐条对照）：
#   1) 样本数 17000
#   2) 单条 input_ids 长度 30~60（不是 128）
#   3) 单条 mask 全 1                <- 单条样本不 padding
#   4) label 类型是 int
#   5) batch 形如 (8, 52)，且每个 batch 的 L 都不一样   <- 动态 padding
#   6) max(real_len) == L            <- 最长的样本决定了这批的宽度
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

MODEL = "bert-base-uncased"
MAX_LEN = 128          # 上限，不是固定长度


class TextDataset(Dataset):
    

    def __init__(self, csv_path: str, tokenizer, max_len: int = MAX_LEN):
        self.df = pd.read_csv(csv_path)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(row["text"], truncation=True, max_length=self.max_len)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(row["target"]),      # 键名是 labels（复数），值必须是 int
        }


def collate_fn(batch, tokenizer):
    """把 N 条不等长样本拼成一个矩阵。

    batch: 长度 N 的列表，每项是 __getitem__ 的返回值
    """
    # 1) 先剥掉 labels —— tokenizer.pad 只处理序列，喂整数会出错
    feats = [{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch]

    # 2) 补到本批最长；BERT 的 pad_token_id 是 0，tokenizer 自己知道
    padded = tokenizer.pad(feats, padding=True, return_tensors="pt")

    # 3) labels 单独堆成张量，dtype 必须是 long（交叉熵的要求）
    padded["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return padded


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = TextDataset("data/split/train.csv", tok)

    print("样本数:", len(ds))

    one = ds[0]
    print("单条 input_ids 长度:", len(one["input_ids"]))
    print("单条 mask 全 1:", all(m == 1 for m in one["attention_mask"]))
    print("label 类型:", type(one["labels"]).__name__)

    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tok),
        num_workers=0,      # Windows 上先设 0
    )

    for step, batch in enumerate(dl):
        real_len = batch["attention_mask"].sum(-1).tolist()   # 1 的个数 = 真实长度
        L = batch["input_ids"].shape[1]
        print(
            f"batch {step}: shape={tuple(batch['input_ids'].shape)} "
            f"L={L} max(real_len)={max(real_len)} real_len={real_len}"
        )
        if step == 2:
            break


if __name__ == "__main__":
    main()
