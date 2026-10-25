"""07_compare_models.py —— 进阶 B：多模型对比实验

运行：.\run 07_compare_models.py

【实验设计】
  控制变量（三个模型完全一致）：
    数据    data/split/train.csv + val.csv（同一份，17000 / 3000）
    类别数  4
    max_len 128
    batch   8
    lr      2e-5
    epochs  1
    seed    0
    bf16    是

  自变量：模型架构
    distilbert-base-uncased   6 层，66M 参数     （蒸馏自 BERT）
    bert-base-uncased        12 层，110M 参数    （你之前的基线）
    roberta-base             12 层，125M 参数    （BERT 的改进版，BPE 词表）

  观测变量（五项）：
    参数量      静态成本，决定模型文件大小
    训练耗时    动态成本，决定迭代速度
    验证准确率  收益
    显存峰值    硬件门槛
    推理吞吐    线上服务的成本（每秒能算多少句）

【必须知道的两个不公平】
  1. 三个模型的分词器不同（roberta 用 BPE，平均 token 数更少），
     所以同样的 max_len 下，实际计算量不完全一样 ->
     耗时对比是"同样配置下的耗时"，不是"纯架构优劣"。
  2. 2e-5 这个学习率对 bert 系友好，对 roberta 不一定最优 ->
     准确率对比是"同样超参下的准确率"，不是各自的天花板。

结论要结合这两点来看，不能简单说"A 比 B 好"。
"""
import csv
import importlib
import time

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]

NUM_LABELS = 4
MAX_LEN = 128
BATCH = 8
LR = 2e-5
EPOCHS = 1
DEV = "cuda"


def load_data_module():
    m = importlib.import_module("02_data")
    return m.TextDataset, m.collate_fn


def build_loader(ds, collate_fn, tok, batch, shuffle):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, tokenizer=tok), num_workers=0,
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    texts = 0
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(DEV) for k, v in batch.items()}
        pred = model(**batch).logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(pred)
        texts += len(pred)
    dt = time.time() - t0
    return correct / max(total, 1), texts / dt


def run_one(name, TextDataset, collate_fn):
    print(f"\n{'=' * 72}\n开始: {name}\n{'=' * 72}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    train_ds = TextDataset("data/split/train.csv", tok, max_len=MAX_LEN)
    val_ds = TextDataset("data/split/val.csv", tok, max_len=MAX_LEN)
    train_loader = build_loader(train_ds, collate_fn, tok, BATCH, True)
    val_loader = build_loader(val_ds, collate_fn, tok, BATCH * 2, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=NUM_LABELS
    ).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params / 1e6:.1f}M   加载耗时: {time.time() - t_load:.0f}s")

    opt = AdamW(model.parameters(), lr=LR)

    # 预热：第一次前向会建 kernel，不能算进耗时
    warm = next(iter(train_loader))
    warm = {k: v.to(DEV) for k, v in warm.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(**warm).loss.backward()
    opt.zero_grad(set_to_none=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} 已用 {time.time() - t0:.0f}s")
        print(f"  epoch {epoch} 平均 loss={running / max(step + 1, 1):.4f}")
    train_time = time.time() - t0

    vram = torch.cuda.max_memory_allocated() / 2**30
    acc, throughput = evaluate(model, val_loader)

    print(f"  结果: acc={acc:.4f}  训练 {train_time:.0f}s  显存 {vram:.2f}GB  "
          f"推理 {throughput:.0f} 句/s")

    del model, opt, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return {
        "模型": name,
        "参数量(M)": round(n_params / 1e6, 1),
        "训练(s)": round(train_time),
        "验证acc": round(acc, 4),
        "显存(GB)": round(vram, 2),
        "推理(句/s)": round(throughput),
    }


def main() -> None:
    TextDataset, collate_fn = load_data_module()

    print(f"配置: max_len={MAX_LEN} batch={BATCH} lr={LR} epochs={EPOCHS} bf16")
    print(f"模型: {', '.join(MODELS)}")

    rows = []
    total_start = time.time()
    for name in MODELS:
        rows.append(run_one(name, TextDataset, collate_fn))

    print(f"\n\n{'=' * 72}")
    print("对比结果")
    print("=" * 72)
    header = ["模型", "参数量(M)", "训练(s)", "验证acc", "显存(GB)", "推理(句/s)"]
    widths = [26, 10, 8, 9, 9, 11]
    print("".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * sum(widths))

    # 以 bert-base 为基线算相对值
    base = next((r for r in rows if r["模型"] == "bert-base-uncased"), rows[0])
    for r in rows:
        line = "".join(str(r[h]).ljust(w) for h, w in zip(header, widths))
        print(line)

    print("-" * sum(widths))
    print("相对 bert-base 基线:")
    for r in rows:
        if r is base:
            continue
        print(f"  {r['模型']}: "
              f"参数量 {r['参数量(M)'] / base['参数量(M)']:.2f}x  "
              f"训练耗时 {r['训练(s)'] / base['训练(s)']:.2f}x  "
              f"准确率差 {r['验证acc'] - base['验证acc']:+.4f}  "
              f"推理速度 {r['推理(句/s)'] / base['推理(句/s)']:.2f}x")

    with open("experiments/model_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已保存: experiments/model_compare.csv")
    print(f"总耗时: {(time.time() - total_start) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
