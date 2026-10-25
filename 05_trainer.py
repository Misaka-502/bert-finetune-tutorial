# 05_trainer.py —— 第 5 课：用 Trainer 重写训练，只为"对照"
#
# 目的：同一套 Dataset / collate_fn，把训练循环换成 transformers 的 Trainer。
#       跑通之后你应该能回答：Trainer 替你写了 03_train.py 里的哪几段？
#
# 为什么不按原教程用 run_classification.py：
#   那份脚本是 transformers 4.x 时代的示例，本机装的是 5.20.1，接口已变，
#   极大概率跑不起来。用 Trainer 直接写反而更短、更透明。
#
# 运行：.\run 05_trainer.py
import importlib
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "bert-base-uncased"
NUM_LABELS = 4
OUT_DIR = "experiments/trainer"


def load_data_module():
    m = importlib.import_module("02_data")
    return m.TextDataset, m.collate_fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main() -> None:
    TextDataset, collate_fn = load_data_module()

    tok = AutoTokenizer.from_pretrained(MODEL)
    train_ds = TextDataset("data/split/train.csv", tok)
    val_ds = TextDataset("data/split/val.csv", tok)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)

    # TODO: 填 TrainingArguments。对照 03_train.py 里的常量，一一对应。
    #   output_dir=OUT_DIR
    #   per_device_train_batch_size=8        <- 对应 03 的 BATCH_SIZE
    #   gradient_accumulation_steps=4        <- 对应 03 的 GRAD_ACCUM
    #   learning_rate=2e-5                   <- 对应 03 的 LR
    #   num_train_epochs=2                   <- 对应 03 的 EPOCHS
    #   bf16=True
    #   eval_strategy="epoch"                <- 旧版 transformers 里叫 evaluation_strategy
    #   save_strategy="no"
    #   logging_steps=50
    #   report_to="none"
    args = None
    if args is None:
        raise NotImplementedError("TODO：构造 TrainingArguments")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, tokenizer=tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
