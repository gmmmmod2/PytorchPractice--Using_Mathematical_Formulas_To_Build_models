# 题目：Batch Normalization

> 目标：按公式实现在批次上的归一化，同时可选仿射。

## 数学定义

设输入 $x \in \mathbb{R}^{B \times C \times \dots}$，其中：

- $B$：batch size（批大小）
- $C$：通道数（feature maps 数）
- $\dots$：额外的空间/序列维度（如高度、高度 × 宽度、序列长度等）
- $m$：在某个通道 $c$ 上，**所有样本在该通道的元素个数**  
  （例如：对于输入 $x \in \mathbb{R}^{B \times C \times H \times W}$, 在通道 $c$ 上 $m = B \times H \times W$）

---

训练阶段（使用当前 batch 统计量）：

$$
\mu_c = \frac{1}{m}\sum_{i=1}^m x_{i,c},\quad
\sigma_c^2 = \frac{1}{m}\sum_{i=1}^m (x_{i,c} - \mu_c)^2
$$

$$
\hat{x}_{i,c} = \frac{x_{i,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

$$
y_{i,c} = \gamma_c \hat{x}_{i,c} + \beta_c
$$

滑动平均缓冲区的更新（PyTorch 约定）：

$$
\mathrm{running\_mean}_c \leftarrow (1-\mathrm{momentum})\cdot \mathrm{running\_mean}_c \;+\; \mathrm{momentum}\cdot \mu_c^{(\text{batch})}
$$

$$
\mathrm{running\_var}_c \leftarrow (1-\mathrm{momentum})\cdot \mathrm{running\_var}_c \;+\; \mathrm{momentum}\cdot (\sigma_c^{2})^{(\text{batch})}
$$

---

推理阶段（使用滑动平均的统计量）：

$$
\hat{x}_{i,c} = \frac{x_{i,c} - \mathrm{running\_mean}_c}{\sqrt{\mathrm{running\_var}_c + \epsilon}}
$$

$$
y_{i,c} = \gamma_c \hat{x}_{i,c} + \beta_c
$$

## 实现要求

- 输入输出形状一致，按通道维度进行归一化；
- 训练时用当前 batch 的均值和方差，推理时用滑动平均；
- 若开启 affine，则对每个通道使用可学习的缩放参数 $\gamma$ 和偏移参数 $\beta$。

## 参考实现（PyTorch）

```python
import torch
import torch.nn as nn

class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps: float = 1e-5, momentum: float = 0.1,
                 elementwise_affine : bool = True, track_running_stats=True):
        super().__init__()
        self.eps = eps

        self.num_features = num_features # 归一化的维度大小
        self.momentum = momentum         # 控制 running_mean / running_var 的更新速率
        self.elementwise_affine = elementwise_affine   # 是否启用放射变换
        self.track_running_stats = track_running_stats # 是否启用滑动平均统计

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x: torch.Tensor):
        dims = [0] + list(range(2, x.dim()))  # 在 batch 和除了 channel 的所有空间维度上统计
        use_batch_stats = self.training or (not self.track_running_stats)  # 不跟踪就始终用 batch 统计

        if use_batch_stats:
            # 模型在训练阶段，或者验证阶段但是滑动平均统计关闭的状态下
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)

            if self.training and self.track_running_stats:
                # 模型在训练阶段, 同时开启了滑动平均就进行记录
                with torch.no_grad():
                    m = self.momentum
                    # 就地更新，保持 buffer 属性与设备/精度
                    self.running_mean.mul_(1 - m).add_(m * mean.squeeze())
                    self.running_var.mul_(1 - m).add_(m * var.squeeze())
        else:
            # 模型在验证阶段且滑动平均统计开启的情况下
            mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var  = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))

        xhat = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            gamma = self.gamma.view(1, -1, *([1] * (x.dim() - 2)))
            beta  = self.beta.view(1, -1, *([1] * (x.dim() - 2)))
            return xhat * gamma + beta
        return xhat
```
