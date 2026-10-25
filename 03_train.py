# 03_train.py —— 第 3 课：训练循环
#
#
# 本机约束（RTX 5070 Laptop 8 GB）：
#   BATCH_SIZE=8 + MAX_LEN=128 + bf16 大约占 3~4 GB，别往上加太多
#
# 验收流程（务必按顺序）：
#   第 1 轮：把 OVERFIT_N 改成 200，EPOCHS 改成 15
#            -> 训练 loss 必须降到 0.1 以下，否则是在排 bug，不是训练
#   第 2 轮：OVERFIT_N 改回 0，EPOCHS 改成 2
#            -> val_acc 应该明显高于 0.25（4 类随机猜）
#
# 出问题先查这三条：
#   loss 变成 None            -> 02_data.py 里键名写成了 label（少了 s）
#   loss 一直卡在 1.38 附近   -> 数据和标签没对上
#   CUDA out of memory        -> BATCH_SIZE 降到 4
import importlib
import time

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "bert-base-uncased"
NUM_LABELS = 4
MAX_LEN = 128
BATCH_SIZE = 8
GRAD_ACCUM = 4          # 等效 batch = 8 * 4 = 32
EPOCHS = 2
LR = 2e-5
OVERFIT_N = 0        # >0 时只取前 N 条做"过拟合测试"
OUT_DIR = "experiments/bert-agnews"


def load_data_module():
    """02_data.py 以数字开头，不能用 `import 02_data`，只能这样加载。"""
    m = importlib.import_module("02_data")
    return m.TextDataset, m.collate_fn


@torch.no_grad()
def evaluate(model, loader):
    """在验证集上算准确率。"""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        pred = model(**batch).logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(pred)
    model.train()
    return correct / max(total, 1)


def main() -> None:
    torch.manual_seed(0)
    TextDataset, collate_fn = load_data_module()

    tok = AutoTokenizer.from_pretrained(MODEL)
    train_ds = TextDataset("data/split/train.csv", tok, max_len=MAX_LEN)
    val_ds = TextDataset("data/split/val.csv", tok, max_len=MAX_LEN)

    if OVERFIT_N:
        train_ds = torch.utils.data.Subset(
            train_ds, range(min(OVERFIT_N, len(train_ds)))
        )
        print(f"[过拟合测试] 只用 {len(train_ds)} 条，{EPOCHS} 轮")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer=tok),
        num_workers=0,      # Windows 上先设 0，开多进程需要 __main__ 保护
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer=tok),
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=NUM_LABELS
    ).to("cuda")

    opt = AdamW(model.parameters(), lr=LR)

    print(
        f"训练样本={len(train_ds)} 验证样本={len(val_ds)} "
        f"batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE * GRAD_ACCUM} "
        f"epochs={EPOCHS} lr={LR}"
    )

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss

            # 梯度累积：除以 GRAD_ACCUM 让梯度变成"平均"而不是"求和"
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            running += loss.item()
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                    f"{time.time() - t0:.0f}s"
                )

        # 收尾：最后不足 GRAD_ACCUM 步的梯度也要更新一次，否则白算
        if len(train_loader) % GRAD_ACCUM != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        acc = evaluate(model, val_loader)
        print(
            f"== epoch {epoch} 平均loss={running / max(step + 1, 1):.4f} "
            f"val_acc={acc:.4f} 用时={time.time() - t0:.0f}s"
        )

    # 只在全量训练时保存；过拟合测试别把废模型存进去
    if OVERFIT_N == 0:
        model.save_pretrained(OUT_DIR)
        tok.save_pretrained(OUT_DIR)
        print(f"已保存到 {OUT_DIR}")
    else:
        print("[过拟合测试] 不保存模型")


if __name__ == "__main__":
    main()
