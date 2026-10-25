# 01_infer.py —— 第 1 课：推理先行
# 目的：把"文本 -> 张量 -> logits -> 概率"这条链看清楚，本节不训练任何东西。
# 运行：& "D:\anaconda\envs\llm\python.exe" 01_infer.py
#
# 学完应能回答：
#   1. input_ids 的 (B, L) 里 L 是什么？
#   2. attention_mask 的 0 代表什么？
#   3. logits 为什么是 (B, num_labels)？
#   4. 二分类能不能用 logits[:, 1] > 0 代替 argmax？
#   5. 为什么必须 eval() + no_grad()？
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME).to(device).eval()
    print(f"词表大小: {tok.vocab_size}   pad_token_id: {tok.pad_token_id}")
    print(f"标签映射: {model.config.id2label}")

    texts = [" I think it isn't worth the time."]

    # ---------- ① 分词：文本 -> token -> id ----------
    print("\n[① 单条分词]")
    enc = tok(texts[0])
    print("tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))
    print("ids   :", enc["input_ids"])

    # ---------- ② 批处理：变长 -> 矩阵 ----------
    batch = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    print("\n[② 批处理]  padding 补到本批最长，不是固定 128")
    print("input_ids      shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    for i, t in enumerate(texts):
        print(f"  句{i + 1} {t!r}")
        print(f"       ids : {batch['input_ids'][i].tolist()}")
        print(f"       mask: {batch['attention_mask'][i].tolist()}")
    print("  ^ 句1 末尾的 0 就是补齐的 [PAD]，mask 里对应位置是 0")

    # ---------- ③ 推理：矩阵 -> logits -> 概率 ----------
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits

    print("\n[③ 模型输出]")
    print("logits shape:", tuple(logits.shape))
    print("logits      :", logits.cpu().numpy().round(4))

    probs = logits.softmax(-1)
    print("softmax(-1) :", probs.cpu().numpy().round(4))
    pred = logits.argmax(-1)
    print("argmax      :", pred.tolist(), " (类别下标，不是标签名)")

    id2label = model.config.id2label
    print("\n最终结果:")
    for t, p, pr in zip(texts, pred.tolist(), probs.cpu().tolist()):
        print(f"  {t!r:48s} -> {id2label[p]:8s} {pr[p]:.4f}")

    # ---------- ④ 常见错误示范：softmax 的维度 ----------
    print("\n[④ softmax 维度对比（别写错）]")
    print("softmax(-1) 每行(每句)求和 =", probs.sum(-1).cpu().numpy().round(4), "  <- 正确")
    print("softmax(0)  每列(每类)求和 =",
          logits.softmax(0).sum(0).cpu().numpy().round(4), "  <- 错误")

    # ---------- ⑤ 二分类的等价写法 ----------
    print("\n[⑤ 二分类可以这样绕开 argmax]")
    print("logits[:, 1] > 0 ->", (logits[:, 1] > 0).tolist(), " 与 argmax 结果一致")
    print("显存占用:", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GB")


if __name__ == "__main__":
    main()
