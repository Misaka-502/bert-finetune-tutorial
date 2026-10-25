"""04_predict_live.py —— 随便写句子，让模型分类（第 4 课的交互版）

两种用法：

  1) 交互式（最常用）：
       .\run 04_predict_live.py
     然后每次输入一句英文，回车看结果；直接回车退出。

  2) 一行命令测多句：
       .\run 04_predict_live.py "Real Madrid won 3-1." "The Fed raised rates."

说明：模型只有 4 个出口（World / Sports / Business / Sci/Tech），
给它这 4 类以外的内容，它也会硬塞进最像的那类，
所以要看"置信度"，而不是只看类别名。
"""
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CKPT = "experiments/bert-agnews"
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def load():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).to("cuda").eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, texts):
    batch = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    return model(**batch).logits.softmax(-1).cpu().tolist()


def show(text, prob):
    order = sorted(range(4), key=lambda i: -prob[i])
    top, second = order[0], order[1]

    print(f"\n  输入: {text}")
    print(f"  预测: {ID2LABEL[top]:9s} {prob[top]:.3f}  {'#' * int(prob[top] * 30)}")
    if prob[second] >= 0.15:
        print(f"  次选: {ID2LABEL[second]:9s} {prob[second]:.3f}")
    if prob[top] < 0.60:
        print("  [低置信度] 这句话不像训练数据里的任何一类，结果不可信")
    print("  四类分布: " + ", ".join(f"{ID2LABEL[i]}={p:.3f}" for i, p in enumerate(prob)))


def main():
    tok, model = load()
    args = sys.argv[1:]

    if args:
        for text, prob in zip(args, predict(tok, model, args)):
            show(text, prob)
        return

    print("输入一句英文，回车分类。直接回车退出。")
    print("试试这些对比例子：")
    print("  Real Madrid beat Barcelona 3-1 in the derby.")
    print("  The Federal Reserve raised interest rates.")
    print("  Apple reported strong earnings as iPhone sales jumped.")
    print("  今天股市大涨，投资者很乐观。")
    print("  The cat sat on the mat.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        show(line, predict(tok, model, [line])[0])


if __name__ == "__main__":
    main()
