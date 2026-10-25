# 04_predict.py —— 第 4 课：加载微调后的模型做预测
#
# 前置：03_train.py 跑完，experiments/bert-agnews/ 里已有模型文件
# 运行：.\run 04_predict.py
#
# 验收：
#   1. 能加载模型 + tokenizer（注意：路径是**目录**，不是模型名）
#   2. 每行概率之和 = 1
#   3. 四句测试句各落到合理的类别
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CKPT = "experiments/bert-agnews"

# ag_news 的类别顺序（来自数据集的 ClassLabel）
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

TEXTS = [
    "A man named John sells his house.",   # 期望 Business
    "Messi scored twice as Barcelona won the final.",       # 期望 Sports
    "NASA launched a new satellite to study the sun.",      # 期望 Sci/Tech
    "The president met with foreign leaders in Geneva.",    # 期望 World
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).to("cuda").eval()

    # 训练时没设置语义标签名，所以 config 里存的是 LABEL_0..3
    print("config 里的标签映射:", model.config.id2label)
    print("我们使用的映射    :", ID2LABEL)
    print()

    batch = tok(TEXTS, padding=True, truncation=True, max_length=128, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    with torch.no_grad():
        logits = model(**batch).logits

    probs = logits.softmax(-1)

    for text, prob in zip(TEXTS, probs.cpu().tolist()):
        best = max(range(len(prob)), key=lambda i: prob[i])
        detail = ", ".join(f"{ID2LABEL[i]}={p:.3f}" for i, p in enumerate(prob))
        print(text)
        print(f"   -> {ID2LABEL[best]:9s} {prob[best]:.4f}")
        print(f"      全部 {{{detail}}}  概率和={sum(prob):.4f}")


if __name__ == "__main__":
    main()
