# app/app.py —— 第 7 课：HuggingFace Spaces 部署版
#
# 部署规则（三条，缺一构建失败）：
#   1. 文件名必须是 app.py，放在 Space 仓库根目录
#   2. 这个目录里的 requirements.txt 不要写 torch 的 cu128 源（Space 没有 GPU）
#   3. 不要 import datasets / accelerate / jupyterlab，部署用不到
#
# 模型有两种来源，二选一：
#   A. 把 experiments/bert-agnews/ 整个目录一起上传，MODEL 写相对路径
#   B. 先把模型 push 到 HuggingFace Hub，MODEL 写仓库 id，比如 "yourname/bert-agnews"
import gradio as gr
from transformers import pipeline

MODEL = "./bert-agnews"          # TODO: 改成你实际上传的路径或 Hub 仓库 id
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Space 免费 CPU 版：device=-1
clf = pipeline("text-classification", model=MODEL, device=-1)


def predict(text: str):
    if not text.strip():
        return {}
    out = clf(text, top_k=4)[0]
    return {
        ID2LABEL[int(r["label"].split("_")[-1])]: r["score"]
        for r in out
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="输入一段新闻"),
    outputs=gr.Label(num_top_classes=4, label="分类结果"),
    title="新闻分类 Demo（ag_news 4 类）",
    description="基于 bert-base-uncased 微调的四类新闻分类器",
    examples=[
        "The stock market rallied today on strong earnings.",
        "Messi scored twice as Barcelona won the final.",
    ],
)

demo.launch()
