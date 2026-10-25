# 从零微调 BERT：4 类新闻分类 → 网页 Demo

用 `transformers` 把 `bert-base-uncased` 微调成一个新闻分类器，再包成浏览器里能点的 Demo。
全部脚本**可直接运行**，按顺序执行即可复现。

| 最终成果 | 数值 |
|---|---|
| 训练数据 | 17000 条（4 类各约 4250） |
| 验证数据 | 3000 条 |
| 验证准确率 | **约 0.93** |
| 模型体积 | 417 MB |
| 训练耗时 | 2 轮，约 10 分钟（RTX 5070 Laptop 8 GB） |
| 显存峰值 | 2.5 ~ 3 GB |

---

## 一、环境要求

| 项目 | 要求 | 说明 |
|---|---|---|
| 系统 | Windows | 启动器是 `.cmd`，其他系统需自己改 |
| GPU | NVIDIA，≥ 8 GB 显存 | 实测峰值 3 GB；用 CPU 也能跑但慢 10~20 倍 |
| Python | 3.11（conda 环境） | 见 [SETUP.md](SETUP.md) |
| 网络 | 能访问 `hf-mirror.com` | 用来下载模型和数据集 |
| 磁盘 | ≥ 5 GB | 模型缓存 1.5 GB + 训练产物 0.5 GB + 依赖 6 GB |

**为什么用启动器而不是直接 `python`：**

```powershell
.\run 01_infer.py      # ✅ 自动处理镜像、缓存、编码、PYTHONPATH
python 01_infer.py     # ❌ 会撞 DNS、重复下载、还可能中文乱码
```

`run.cmd` 是一个 6 行的批处理，它做的事：清空 `PYTHONPATH`、设 `HF_ENDPOINT`、把模型缓存指到本项目、强制 UTF-8、调用 conda 环境里的 python。

---

## 二、快速开始（5 步）

```powershell
# 0) 进项目目录
cd <本项目路径>

# 1) 体检：软件栈 + GPU 架构（必须看到 (12, 0) 和 matmul OK）
.\run check_env.py
.\run 00_offline_test.py

# 2) 下载数据（ag_news → data/split/*.csv，约 20 秒）
.\run 02_make_data.py

# 3) 训练（2 轮，约 10 分钟；模型存到 experiments/bert-agnews/）
.\run 03_train.py

# 4) 预测
.\run 04_predict.py

# 5) 网页 Demo（浏览器打开 http://127.0.0.1:7860）
.\run 06_gradio.py
```

跑完 5 步，你会得到 `experiments/bert-agnews/` 里 4 个文件（模型 + tokenizer），以及一个能打字的网页。

---

## 三、课程地图

| 步骤 | 文件 | 学什么 | 耗时 | 验收标准 |
|---|---|---|---|---|
| 体检 | `check_env.py` | 环境是否健康 | 1 min | `(12, 0)` + `matmul OK` |
| 体检 | `00_offline_test.py` | 不联网验证软件栈 | 1 min | `反向传播: OK` |
| 第 1 课 | `01_infer.py` | 文本 → 张量 → logits → 概率 | 40 min | `(2, 12)` / `(2, 2)` / `POSITIVE 0.99` |
| 第 2 课·前 | `02_make_data.py` | 数据集 → csv | 5 min | `train: 17000  val: 3000` |
| 第 2 课 | `02_data.py` | `Dataset` + `collate_fn` + 动态 padding | 1.5 h | batch 形如 `(8, 52)`，**每个 batch 的 L 都不同** |
| 第 3 课 | `03_train.py` | 训练循环、梯度累积、bf16 | 1.5 h | 起步 loss≈1.38，2 轮后 val_acc > 0.9 |
| 第 4 课 | `04_predict.py` | 加载自训模型、读置信度 | 40 min | 4 句测试句各落对类别 |
| 第 4 课·增强 | `04_predict_live.py` | 交互式预测 | 20 min | 随意输入句子能出概率 |
| 第 5 课（可选） | `05_trainer.py` | Trainer 对照手写循环 | 1 h | 能说出 Trainer 省了哪 4 段代码 |
| 第 6 课 | `06_gradio.py` | 本地网页 Demo | 1 h | 7860 端口能出 4 类概率条 |
| 第 7 课（待条件） | `app/` | 部署到 Spaces | — | 需能访问 `huggingface.co` |
| 进阶 B | `07_compare_models.py` | 多模型对比实验设计 | 1 h | 产出「模型-耗时-准确率」对比表 |
| 进阶 C | `08_lora.py` | LoRA 微调（只训 0.27% 参数） | 1 h | adapter 1.1 MB，重载/合并输出一致 |

**建议顺序**：先按 1→2→3→4→6 跑通主线，再回头看 5、7 和进阶。

---


## 四、目录结构

```
.
├── run.cmd                    启动器（必须用它跑脚本）
├── hf.cmd                     用 hf CLI 下载模型
├── requirements.txt           依赖清单（不含 torch）
├── SETUP.md                   环境搭建记录 + 踩坑清单
├── check_env.py               环境体检
├── 00_offline_test.py         离线验证软件栈
├── 01_infer.py                第 1 课：推理
├── 02_make_data.py            第 2 课·前：数据集 → csv
├── 02_data.py                 第 2 课：数据管道
├── 03_train.py                第 3 课：训练
├── 04_predict.py              第 4 课：预测
├── 04_predict_live.py         第 4 课：交互式预测
├── 05_trainer.py              第 5 课：Trainer 对照
├── 06_gradio.py               第 6 课：网页 Demo
├── 07_compare_models.py       进阶 B：多模型对比
├── 08_lora.py                 进阶 C：LoRA
├── app/                       第 7 课：部署到 Spaces
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── raw/                   原始数据（只读）
│   └── split/                 train.csv / val.csv（由脚本生成）
├── experiments/               训练产物（模型 + tokenizer）
└── .cache/hf/                 模型和数据集缓存（gitignore，约 1.5 GB）
```

---

## 五、常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `No module named 'torch'` | 用了系统 Python，不是 conda 环境 | 用 `.\run`，别用 `python` |
| `Cannot find empty port 7860` | 上一个 Gradio 还在跑 | `Ctrl+C` 关掉，或改 `server_port` |
| `getaddrinfo failed` | 没走启动器，镜像没生效 | 用 `.\run` |
| `Column 'train' doesn't exist` | `load_dataset(split=...)` 后又 `ds["train"]` | 写了 `split=` 就直接 `ds.to_pandas()` |
| `loss` 是 `None` | `02_data.py` 里键名写成了 `label` | 改成 `labels`（复数） |
| `CUDA out of memory` | batch 太大 | `BATCH_SIZE` 从 8 降到 4 |
| `(12, 0)` 打印不出来 | torch 装成了旧架构版本 | 见 SETUP.md 第一节，重装 cu128 |
| `OSError: ... model.safetensors not found` | 还没训练 | 先跑 `03_train.py` |

**排查顺序**：`check_env.py` → `00_offline_test.py` → 对应课程的脚本。

---

## 六、进阶路线

| 方向 | 怎么做 |
|---|---|
| 换数据集 | 改 `02_make_data.py` 里 `load_dataset(...)` 那一行，再改 `03_train.py` 的 `NUM_LABELS` |
| 换模型 | 改 `03_train.py` 的 `MODEL` 常量（tokenizer 和 model 必须同时换） |
| 换语言 | `bert-base-chinese` + 中文数据集，**模型和数据必须一起换** |
| 换任务类型 | 问答/摘要需要改 Dataset 和模型 head，参考 `transformers/examples/pytorch/` |
| 大模型 + LoRA | 7B 级需要 `bitsandbytes` 4bit 量化，8 GB 显存勉强能跑 7B |

**换数据的最小改动**：

```python
# 02_make_data.py 第 9 行附近
ds = load_dataset("stanfordnlp/imdb", split="train")     # 换成电影评论
df = ds.to_pandas()[["text", "label"]]
```

```python
# 03_train.py
NUM_LABELS = 2        # imdb 是二分类
```

其余代码一行不用改——这是第 2 课"数据管道与模型解耦"的直接体现。



