# W4A16 GEMM CUDA 融合算子

本算子将 W4A16 权重量化矩阵乘法中的**反量化（dequantization）与 GEMM 计算**融合为单一 CUDA 内核。相比先反量化再计算的传统方式，融合内核直接在 GPU 上完成 INT4→FP16 转换与矩阵乘法，避免中间结果回写显存，显著减少内存带宽访问，从而加速大语言模型的推理过程并减少显存占用。

## 编译

```bash
python setup.py build_ext --inplace
```

## 环境配置（Linux/WSL）

Linux/WSL 环境需设置动态库搜索路径：

```python
import os
os.environ['LD_LIBRARY_PATH'] = (
    '/path/to/conda/env/lib/python3.10/site-packages/torch/lib:'
    '/usr/local/cuda/lib64'
)
```

或者在终端执行：

```bash
export LD_LIBRARY_PATH=/path/to/conda/env/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Windows 环境无需此配置。

## 完整调用示例

```python
import torch
import w4a16_gemm

# 参数设置
batch_size = 1
in_features = 1024
out_features = 4096
group_size = 128

num_groups = in_features // group_size
padded_in_features = (in_features + 1) // 2 * 2

# 准备输入
input_fp16 = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')

# 准备权重 (INT4 打包)
weight_int4 = torch.randint(0, 256, (out_features, padded_in_features // 2), dtype=torch.uint8, device='cuda')

# 准备缩放因子
scale = torch.rand(out_features, num_groups, dtype=torch.float16, device='cuda')

# 准备偏置 (可选)
bias = torch.randn(out_features, dtype=torch.float16, device='cuda')

# 调用 W4A16 GEMM
output = w4a16_gemm.w4a16_gemm(
    weight_int4,
    scale,
    input_fp16,
    bias,
    in_features
)

print(f"输出形状: {output.shape}")  # (1, 4096)
```

## 参数说明

| 参数 | 类型 | 形状 | 描述 |
|------|------|------|------|
| weight_int4 | torch.Tensor (uint8) | (out_features, padded_in_features//2) | 量化权重 |
| scale | torch.Tensor (float16) | (out_features, num_groups) | 缩放因子 |
| input | torch.Tensor (float16) | (batch_size, in_features) | 输入张量 |
| bias | torch.Tensor (float16) | (out_features,) | 偏置 |
| in_features | int | - | 输入维度 |

## 返回

- output: torch.Tensor (float16), shape: (batch_size, out_features)

## 环境要求

- CUDA Toolkit 11.0+
- PyTorch 2.0+
- GPU 计算能力 7.0+

## 许可证

MIT
