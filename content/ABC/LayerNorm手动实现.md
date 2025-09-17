# 题目：Layer Normalization(不含仿射/含仿射两版)

> 目标 从公式实现 LayerNorm。

## 数学定义

输入 $x \in \mathbb{R}^{B \times L \times d}$，对最后两维或一维做归一化：

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

if __name__=='__main__':
    x=torch.randn(2,8,16)
    #对最后一维进行LN
    last_one_dim_LN=MyLayerNorm(normalized_shape=(16,),elementwise_affine=True)
    result=last_one_dim_LN(x)
    print(result.shape)
    print("-"*60)
    # 对最后二维进行LN
    last_one_dim_LN = MyLayerNorm(normalized_shape=(8,16), elementwise_affine=True)
    result = last_one_dim_LN(x)
    print(result.shape)
```
