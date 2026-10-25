# 06_gradio.py —— 第 6 课：本地 Gradio Demo
#
# 前置：experiments/bert-agnews/ 里有模型（03_train.py 的产物）
# 运行：c
# 然后浏览器打开 http://127.0.0.1:7861
#
# 验收：网页里输入一句话 -> 出现 4 个类别的概率条，改文本结果会变
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

CKPT = "experiments/bert-agnews"
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def build():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT)

    # 关键一步：把语义标签写进 config。
    # 训练时只写了 num_labels=4，config 里存的是 LABEL_0..3，
    # 不改的话 pipeline 会返回 "LABEL_2" 这种没有意义的名字。
    model.config.id2label = ID2LABEL
    model.config.label2id = {v: k for k, v in ID2LABEL.items()}

    clf = pipeline("text-classification", model=model, tokenizer=tok, device=0)

    def predict(text: str):
        if not text or not text.strip():
            return {}
        out = clf(text, top_k=4)
        if out and isinstance(out[0], list):   # 传列表时返回的是嵌套列表
            out = out[0]
        return {r["label"]: float(r["score"]) for r in out}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=4, label="输入一段新闻"),
        outputs=gr.Label(num_top_classes=4, label="分类结果"),
        title="新闻分类 Demo（ag_news 4 类）",
        description="bert-base-uncased 微调。置信度低说明这句话不像训练数据里的任何一类。",
        examples=[
            "The stock market rallied today on strong earnings.",
            "Messi scored twice as Barcelona won the final.",
            "NASA launched a new satellite to study the sun.",
            "The president met with foreign leaders in Geneva.",
        ],
    )
    demo.launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    build()
