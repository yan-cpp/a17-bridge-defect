# Environment Setup

## Recommended Baseline

- Windows 10 or Windows 11
- Python 3.11
- NVIDIA GPU with at least 12GB VRAM
- Recommend RTX 4090 24GB for A17 4K work
- CUDA-compatible PyTorch and NVIDIA driver

## Create Virtual Environment

```powershell
cd D:\zm\newA\a17-bridge-defect
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Install Base Dependencies

```powershell
pip install -r requirements-dev.txt
```

## Install PyTorch

按本机CUDA版本使用PyTorch官网给出的安装命令。示例：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Install Machine Learning Dependencies

```powershell
pip install -r requirements-ml.txt
```

## Verify Environment

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m a17 doctor
python -m a17 validate-config
python -m a17 smoke
python -m unittest discover -s tests -v
```

## Expected Doctor Output

`doctor` 会输出：

- Python版本和执行文件。
- 当前工作目录。
- 配置文件校验结果。
- 可选依赖安装状态。
- GPU型号、显存和驱动版本，若`nvidia-smi`可用。

缺少模型依赖时，基础配置校验和单元测试仍应正常运行。`doctor --strict` 会在缺少依赖时返回失败。

## Current Machine Note

项目初始环境检查检测到 NVIDIA GeForce RTX 5060 Laptop GPU，显存为 8151 MiB。该配置可以用于配置校验、小尺寸切块实验和轻量训练，但不足以直接进行大批量4K整图训练。

当前机器建议：

- 使用1024x1024切块。
- Batch Size先从1开始。
- 开启混合精度。
- 必要时使用梯度累积。
- 4K整图和最终模型训练优先使用24GB显存GPU。
- 不要在8GB显存机器上强行设置较大的Batch Size。
