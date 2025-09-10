# 题目：Layer Normalization（不含仿射/含仿射两版）

**目标**：从公式实现 LayerNorm，并比较禁用与启用可学习仿射的差别。

---

## 数学定义

对输入 $x \in \mathbb{R}^{B \times L \times d}$，对最后一维做归一化：

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

---

## 实现要求

- 自行实现 `LayerNorm`，支持 `elementwise_affine=True/False`。
- 与 `torch.nn.LayerNorm` 数值比对。

---

## 参考实现

```python
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        # 归一化最后一维
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            return self.gamma * xhat + self.beta
        return xhat

# 数值对比
x = torch.randn(3, 4, 8)
m1 = MyLayerNorm(8, elementwise_affine=False)
ref1 = nn.LayerNorm(8, elementwise_affine=False)
print(torch.allclose(m1(x), ref1(x), atol=1e-6))

m2 = MyLayerNorm(8, elementwise_affine=True)
ref2 = nn.LayerNorm(8, elementwise_affine=True)
# 同步参数验证
with torch.no_grad():
    ref2.weight.copy_(m2.gamma)
    ref2.bias.copy_(m2.beta)
print(torch.allclose(m2(x), ref2(x), atol=1e-6))
```
