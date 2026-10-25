# 环境记录（SETUP）

本机环境是怎么搭起来的、踩过哪些坑、以后怎么复现。出问题时先看这份。

---

## 一、最终环境

| 项目 | 值 |
|---|---|
| 显卡 | NVIDIA GeForce RTX 5070 Laptop GPU（8 GB，Blackwell / sm_120） |
| 驱动 | 617.42 |
| conda | 26.1.1，base 在 `D:\anaconda` |
| 环境路径 | `D:\anaconda\envs\llm` |
| Python | 3.11.17 |
| torch | **2.11.0+cu128**（CUDA 12.8，capability `(12, 0)`） |
| numpy | 2.4.6 |
| transformers | 5.20.1 |
| datasets | 5.1.0 |
| evaluate | 0.4.6 |
| accelerate | 1.15.0 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| gradio | 6.30.0 |

`pip check` → `No broken requirements found.`

---

## 二、为什么不能用教程的 requirements.txt

| 冲突 | 教程 | 本机现实 |
|---|---|---|
| torch 版本 | `torch==2.0.0` | RTX 5070 是 sm_120，2.0 的 wheel 没有该架构 kernel，必须 cu128 且 ≥2.7 |
| Python | 3.9 | 已 EOL，很多新 wheel 不再提供 |
| 安装方式 | `pip install -r requirements.txt` | 会装 CPU 版 torch，并覆盖已装好的 GPU 版 |

---

## 三、必须记住的四条命令

```powershell
# 1. 装 torch（唯一需要指定 cu128 源的包）
& "D:\anaconda\envs\llm\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 装其余依赖（这条里不能出现 torch）
& "D:\anaconda\envs\llm\python.exe" -m pip install transformers datasets evaluate accelerate scikit-learn pandas gradio

# 3. 验证 GPU（capability 必须是 (12, 0)）
& "D:\anaconda\envs\llm\python.exe" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"

# 4. 体检
.\run check_env.py
```

---

## 四、踩过的坑（按严重程度）

### 坑 1：`PYTHONPATH=D:\opencv` 污染全局

**症状**：`import torch` 报 `No module named 'numpy._core._multiarray_umath'`，且 traceback 里 numpy 路径是 `D:\opencv\numpy`。

**原因**：用户级 `PYTHONPATH` 指向 `D:\opencv`，里面的 numpy 是给 Python 3.12 编的（`cp312`），而环境是 3.11，C 扩展 ABI 不兼容。**环境里装了正确的 numpy 也永远轮不到它被加载。**

**修复**（已做）：

```powershell
[Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "User")
```

⚠️ 修改注册表只影响**之后启动**的进程，已开的窗口必须重开才生效。

**副作用**：`D:\opencv` 那套 OpenCV 如果需要 `PYTHONPATH` 才能 import cv2，现在会失效。需要时临时加：

```powershell
$env:PYTHONPATH = "D:\opencv"      # 只对当前窗口
```

### 坑 2：`huggingface.co` 无法解析

**症状**：`getaddrinfo failed` / `DNS 服务器失败`。`hf-mirror.com` 可解析、443 可通。

**修复**：`run.cmd` 里已设 `HF_ENDPOINT=https://hf-mirror.com`。

注意镜像会**限流（HTTP 429）**，重试即可；huggingface_hub 自带退避。

### 坑 3：PowerShell 禁止运行脚本

**症状**：`& "D:\anaconda\shell\condabin\conda-hook.ps1"` 报"在此系统上禁止运行脚本"。

**修复**（可选，目前用 `run.cmd` 绕开了）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
conda init powershell
```

顺序不能反：先改策略，再 init。

### 坑 4：HF 缓存默认用符号链接

**症状**：下载时报 `Permission denied ... dummy_file_src`。

**修复**：`run.cmd` 里已设 `HF_HUB_DISABLE_SYMLINKS=1`，改为复制文件。

### 坑 5：`load_dataset(..., split="train")` 后再取 `ds["train"]`

**症状**：`ValueError: Column 'train' doesn't exist.`

**原因**：`DatasetDict["train"]` 是取 split，`Dataset["train"]` 是取**列**。加了 `split=` 就已经是 `Dataset` 了。

**规则**：写了 `split=` 就直接 `ds.to_pandas()`；不写才用 `ds["train"]`。

---

## 五、`run.cmd` 内容（启动器）

```bat
@echo off
REM run.cmd - launcher for this workspace. usage: .\run 01_infer.py
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%~dp0.cache\hf"
set "HF_HUB_DISABLE_SYMLINKS=1"
"D:\anaconda\envs\llm\python.exe" %*
```

注意：`.cmd` 文件里**不要写中文注释**，cmd 按 OEM 代码页读文件，中文会被误当成命令执行。

---

## 六、缓存位置

```
transformer\.cache\hf\
├── hub\             模型（distilbert 256 MB，bert-base 约 440 MB）
└── datasets\        数据集（ag_news 约 20 MB）
```

默认位置 `C:\Users\misaka\.cache\huggingface` **是空的**——因为我们用 `HF_HOME` 改到了工作区。

**用 `.\run` 跑就在工作区，直接调 python 就在默认位置，两边互不相通，会重复下载。**

---

## 七、重建这套环境的完整顺序

```powershell
conda create -n llm python=3.11 -y
conda run -n llm python -V                     # 确认 3.11

# 清污染
[Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "User")

# torch 先装，且只用 cu128 源
& "D:\anaconda\envs\llm\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& "D:\anaconda\envs\llm\python.exe" -c "import torch; print(torch.cuda.get_device_capability(0))"   # (12, 0)

# numpy 必须单独装（torch 不依赖它，但 transformers 要）
& "D:\anaconda\envs\llm\python.exe" -m pip install numpy

# 其余依赖
& "D:\anaconda\envs\llm\python.exe" -m pip install transformers datasets evaluate accelerate scikit-learn pandas gradio

& "D:\anaconda\envs\llm\python.exe" -m pip check
& "D:\anaconda\envs\llm\python.exe" -m pip freeze > requirements.lock.txt
```

---

## 八、8 GB 显存的现实约束

| 场景 | 配置 |
|---|---|
| 推理 distilbert | 0.26 GB |
| 训练 bert-base（batch=8, len=128, bf16） | 约 3~4 GB |
| 训练 bert-base（batch=32, len=512） | 约 20 GB ❌ 必爆 |

所以本项目的固定配置是：`BATCH_SIZE=8` + `MAX_LEN=128` + `bf16` + 梯度累积 4（等效 batch 32）。

新闻/推文类文本平均 30~40 个 token，`MAX_LEN=128` 完全够，512 是纯浪费。
