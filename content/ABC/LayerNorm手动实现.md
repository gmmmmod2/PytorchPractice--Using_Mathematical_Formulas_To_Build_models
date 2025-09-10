# 题目：Layer Normalization（不含仿射/含仿射两版）

**目标**：从公式实现 LayerNorm。

## 数学定义

输入 $x \in \mathbb{R}^{B \times L \times d}$，对最后一维做归一化：

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,\quad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2
$$

$$
\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

带仿射参数时，输出：

$$
y = \gamma \odot \mathrm{LN}(x) + \beta,\quad \gamma,\beta \in \mathbb{R}^{d}
$$

## 实现要求

- 自行实现 `LayerNorm`，支持 `elementwise_affine=True/False`。

## 参考实现

```python
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5, elementwise_affine: bool=True):
        super().__init__()
        self.normalized_shape = normalized_shape     # 表示需要放射变换部分的维度
        self.elementwise_affine = elementwise_affine # 是否开启放射变换计算
        self.eps = eps
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            # 保持模型结构一致
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            xhat = self.gamma * xhat + self.beta
        return xhat
```
