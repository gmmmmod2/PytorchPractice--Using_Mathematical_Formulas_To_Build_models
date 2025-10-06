# 题目：Layer Normalization(多维归一化版)

> 目标：按公式实现对**最后一维或最后两维可选**的归一化方式，同时可选仿射。

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
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5, elementwise_affine: bool=True):
        super(MyLayerNorm, self).__init__()
        self.eps=eps
        self.elementwise_affine=elementwise_affine # 是否开启放射变换计算
        self.normalized_shape=normalized_shape # 表示需要放射变换部分的维度
        if elementwise_affine:
            self.gamma=nn.Parameter(torch.ones(normalized_shape))
            self.beta=nn.Parameter(torch.zeros(normalized_shape))
        else:
            # 保持模型结构一致
            self.register_parameter('gamma',None)
            self.register_parameter('beta',None)

    def forward(self, x: torch.Tensor):
        dims=[-(i+1) for i in range(len(self.normalized_shape))]
        mean=x.mean(dim=dims,keepdim=True)
        var=x.var(dim=dims,unbiased=False,keepdim=True)
        xhat=(x-mean)/torch.sqrt(var+self.eps)
        if self.elementwise_affine:
            return xhat*self.gamma+self.beta
        else:
            return xhat
```
