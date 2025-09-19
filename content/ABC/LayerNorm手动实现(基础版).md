# 题目：Layer Normalization(一维版)

> 目标：按公式实现对**最后一维**的归一化，同时可选仿射。

## 数学定义

输入 $x \in \mathbb{R}^{B \times L \times d}$：

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,\quad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2
$$

$$
\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

若使用仿射变换：

$$
y = \gamma \odot \mathrm{LN}(x) + \beta,\quad \gamma,\beta \in \mathbb{R}^{d}
$$

## 实现要求

- 自行实现 `LayerNorm`，支持 `elementwise_affine=True/False`。

## 参考实现（PyTorch）

```python
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        x: (B,L,d)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            return xhat * self.gamma + self.beta

        return xhat
```
