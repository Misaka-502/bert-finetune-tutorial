"""08_lora.py —— 进阶 C：LoRA 微调（只训 0.27% 的参数）

运行：.\run 08_lora.py

【本节课要回答的问题】
  1. 全量微调 1.1 亿参数，LoRA 只训 30 万，效果差多少？
  2. r（秩）调大调小，参数量怎么变？
  3. LoRA 的产物有多大？怎么加载？怎么合并回普通模型？

【核心原理一句话】
  冻结原模型所有参数，只在部分层旁边挂两个小矩阵 A、B，
  训练时只更新 A、B，前向时把 B@A 加回原来的权重上：
      W' = W + (alpha / r) * B @ A
  B@A 的形状和 W 一样，但秩最多是 r（r 通常 4~64），
  所以可训练参数从 768x768 降到 r x 768 x 2。

【和全量微调的对比要公平】
  LoRA 的合理学习率是 1e-4 量级，全量微调是 2e-5 量级。
  这不是"随便改的"，而是因为 LoRA 训的是新加的小矩阵，
  它们从随机初始化开始，需要更大的步子。
  所以下面这张对比表里 lr 不同，必须在结论里说明。
"""
import importlib
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "bert-base-uncased"
NUM_LABELS = 4
MAX_LEN = 128
BATCH = 8
LR_LORA = 1e-4          # LoRA 用 1e-4（全量微调是 2e-5）
EPOCHS = 1
DEV = "cuda"
ADAPTER_DIR = "experiments/lora-bert-agnews"
TARGETS = ["query", "value"]   # 给哪些层挂 adapter

ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
TEXTS = [
    "The stock market rallied today on strong earnings.",
    "Messi scored twice as Barcelona won the final.",
    "NASA launched a new satellite to study the sun.",
    "The president met with foreign leaders in Geneva.",
]


def load_data_module():
    m = importlib.import_module("02_data")
    return m.TextDataset, m.collate_fn


def make_lora(rank):
    """造一个挂了 LoRA 的模型。"""
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,      # 缩放系数，惯例取 2r
        lora_dropout=0.1,
        target_modules=TARGETS,
        task_type="SEQ_CLS",      # 告诉 peft 这是分类任务，分类头也要训
    )
    return get_peft_model(base, cfg)


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def predict(model, tok, texts):
    model.eval()
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    batch = {k: v.to(DEV) for k, v in batch.items()}
    return model(**batch).logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(DEV) for k, v in batch.items()}
        pred = model(**batch).logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(pred)
    model.train()
    return correct / max(total, 1)


def scan_ranks():
    print("=" * 72)
    print("一、r 扫描（只数参数，不训练）")
    print("=" * 72)

    total = sum(p.numel() for p in
                AutoModelForSequenceClassification.from_pretrained(
                    MODEL, num_labels=NUM_LABELS).parameters())
    torch.cuda.empty_cache()

    print(f"  全量模型总参数: {total:,}")
    print(f"  {'r':>3}  {'可训练参数':>12}  {'占比':>8}  {'adapter 理论大小':>16}")
    print("  " + "-" * 46)
    for rank in (2, 4, 8, 16, 32):
        m = make_lora(rank)
        _, trainable = count(m)
        # adapter 是 fp32，每个参数 4 字节
        print(f"  {rank:>3}  {trainable:>12,}  {trainable / total:>7.3%}  "
              f"{trainable * 4 / 1024 / 1024:>14.2f} MB")
        del m
        torch.cuda.empty_cache()
    print()


def train_lora(TextDataset, collate_fn, tok):
    print("=" * 72)
    print("二、训练 LoRA（r=8, lr=1e-4, 1 轮）")
    print("=" * 72)

    torch.manual_seed(0)
    train_ds = TextDataset("data/split/train.csv", tok, max_len=MAX_LEN)
    val_ds = TextDataset("data/split/val.csv", tok, max_len=MAX_LEN)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer=tok), num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH * 2, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer=tok), num_workers=0)

    model = make_lora(8).to(DEV)
    model.print_trainable_parameters()

    # 关键：优化器只拿"需要梯度的参数"
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_LORA)

    torch.cuda.reset_peak_memory_stats()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()

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
            print(f"  step {step}/{len(train_loader)} loss={loss.item():.4f}")

    t1.record()
    torch.cuda.synchronize()
    train_sec = t0.elapsed_time(t1) / 1000

    vram = torch.cuda.max_memory_allocated() / 2**30
    acc = evaluate(model, val_loader)
    print(f"\n  平均 loss={running / max(step + 1, 1):.4f}  "
          f"val_acc={acc:.4f}  训练 {train_sec:.0f}s  显存 {vram:.2f}GB")
    return model, tok, acc, train_sec, vram


def save_and_reload(model, tok):
    print("\n" + "=" * 72)
    print("三、保存 adapter（只有几 MB），再加载回来验证")
    print("=" * 72)

    model.save_pretrained(ADAPTER_DIR)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(ADAPTER_DIR) for f in fs)
    print(f"  adapter 目录: {ADAPTER_DIR}")
    for f in sorted(os.listdir(ADAPTER_DIR)):
        print(f"    {f:40s} {os.path.getsize(os.path.join(ADAPTER_DIR, f)) / 1024:>8.1f} KB")
    print(f"  总大小: {size / 1024 / 1024:.2f} MB"
          f"   （全量模型是 440 MB，约 {440 / (size / 1024 / 1024):.0f} 分之一）")

    # 加载：先复原 base，再挂上 adapter
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
    reloaded = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEV)

    before = predict(model, tok, TEXTS)
    after = predict(reloaded, tok, TEXTS)
    same = torch.allclose(before, after, atol=1e-5)
    print(f"\n  重载后输出一致: {same}  （最大差异 "
          f"{(before - after).abs().max().item():.2e}）")
    return reloaded


def merge_and_check(model, tok):
    print("\n" + "=" * 72)
    print("四、合并 adapter 回主权重（部署用）")
    print("=" * 72)
    before = predict(model, tok, TEXTS)
    merged = model.merge_and_unload().to(DEV)
    after = predict(merged, tok, TEXTS)
    print(f"  合并后输出一致: {torch.allclose(before, after, atol=1e-4)}  （最大差异 "
          f"{(before - after).abs().max().item():.2e}）")
    print(f"  合并后参数量: {sum(p.numel() for p in merged.parameters()):,}  "
          f"（已是普通 BERT，不再需要 peft）")
    print("  合并的意义：推理时没有额外开销，可以直接用 AutoModel 加载，")
    print("              适合最终部署；不合并则要同时保留 base + adapter。")
    return merged


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    TextDataset, collate_fn = load_data_module()

    scan_ranks()
    model, tok, acc, sec, vram = train_lora(TextDataset, collate_fn, tok)
    model = save_and_reload(model, tok)
    merge_and_check(model, tok)

    print("\n" + "=" * 72)
    print("五、和全量微调对比")
    print("=" * 72)
    print(f"  {'方案':<14}{'可训练参数':>12}{'占比':>9}{'训练(s)':>10}{'显存(GB)':>10}{'val_acc':>9}")
    print("  " + "-" * 62)
    print(f"  {'LoRA r=8':<14}{297988:>12,}{297988 / 109783304:>8.3%}"
          f"{sec:>10.0f}{vram:>10.2f}{acc:>9.4f}")
    try:
        import csv
        with open("experiments/model_compare.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["模型"] == "bert-base-uncased":
                    print(f"  {'全量微调':<14}{'109.8M':>12}{'100.0%':>9}"
                          f"{row['训练(s)']:>10}{row['显存(GB)']:>10}{row['验证acc']:>9}")
    except FileNotFoundError:
        print("  （没找到 model_compare.csv，先跑 07_compare_models.py 才有全量基线）")

    print("\n  结论要点：")
    print("    - 参数量差 369 倍，效果差距通常 1~3 个百分点")
    print("    - 显存主要花在冻结的主干上，LoRA 省的是优化器状态")
    print("    - 数据量越大、类别越多，LoRA 的优势越明显")
    print("    - lr 不同（1e-4 vs 2e-5），对比时必须在结论中说明")


if __name__ == "__main__":
    main()
