# check_env.py —— 环境体检脚本
# 运行：& "D:\anaconda\envs\llm\python.exe" check_env.py
import sys


def main() -> None:
    print("=" * 60)
    print("解释器:", sys.executable)
    print("版本  :", sys.version.split()[0])
    print("=" * 60)

    if "envs\\llm" not in sys.executable:
        print("[警告] 当前不在 llm 环境里，结果不可信")

    bad = [p for p in sys.path if "opencv" in p.lower()]
    print("sys.path 含 opencv:", bad if bad else "无（正常）")

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print("[FAIL] numpy 导入失败:", e)
        return
    print("\nnumpy :", np.__version__)
    print("        ", np.__file__)

    import torch

    print("\ntorch        :", torch.__version__)
    print("cuda build   :", torch.version.cuda)
    print("cuda 可用    :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("[FAIL] GPU 不可用，请检查驱动")
        return

    print("设备         :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("计算能力     :", cap, "<- 必须是 (12, 0)")

    a = torch.randn(4096, 4096, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print("矩阵乘       : OK", round(b.float().mean().item(), 4))
    print("显存         :", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")

    print("\n--- 以下缺失不影响第 1 课 ---")
    for m in ("transformers", "datasets", "evaluate", "accelerate", "sklearn", "pandas", "gradio"):
        try:
            mod = __import__(m)
            print(f"{m:14s}: {getattr(mod, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"{m:14s}: 未安装 ({e})")


if __name__ == "__main__":
    main()
