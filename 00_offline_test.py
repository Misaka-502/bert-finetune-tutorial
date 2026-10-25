# 00_offline_test.py —— 不联网验证 transformers + torch 能正常前向/反向
# 用途：模型下载被墙或网络不稳时，先用这个确认"软件栈"没问题
import torch
from transformers import BertConfig, BertForSequenceClassification

def main() -> None:
    print("torch:", torch.__version__, "| cuda:", torch.cuda.get_device_capability(0))

    cfg = BertConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=4,
    )
    model = BertForSequenceClassification(cfg).to("cuda")

    # 造一个假 batch：2 条句子、每条 16 个 token
    ids = torch.randint(0, 30522, (2, 16), device="cuda")
    mask = torch.ones_like(ids)
    labels = torch.tensor([0, 3], device="cuda")

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    print("前向 logits:", tuple(logits.shape), "<- 应为 (2, 4)")

    model.train()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    print("loss:", round(out.loss.item(), 4), "<- 4 分类随机猜测约 1.386")
    out.loss.backward()
    print("反向传播: OK")
    print("峰值显存:", round(torch.cuda.max_memory_allocated() / 2**20, 1), "MB")


if __name__ == "__main__":
    main()
