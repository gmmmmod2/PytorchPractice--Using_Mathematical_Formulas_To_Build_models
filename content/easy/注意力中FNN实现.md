# 题目：Position-wise 前馈网络（FeedForward Layer）实现

> 目标：实现 Transformer 中的前馈子层（Position-wise FeedForward, FFN），加深对 Transformer 结构的理解，并熟悉 PyTorch 基本模块搭建。

## 数学定义

对输入序列表示 $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$，FFN 独立作用在每个位置（不共享时序信息），计算过程为：

$$
\text{FFN}(X) = \max\!\big(0,\, X W_1 + b_1\big) W_2 + b_2
$$

其中：

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，$b_1 \in \mathbb{R}^{d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$b_2 \in \mathbb{R}^{d_{\text{model}}}$
- $\max(0, \cdot)$ 表示 ReLU 激活
- $d_{\text{ff}}$ 通常比 $d_{\text{model}}$ 大，例如 $d_{\text{ff}}=4d_{\text{model}}$

## 实现要求

1. 输入张量形状为 `(B, T, d_model)`。
2. 实现两个线性变换，中间加入 ReLU 激活。
3. 支持自定义 `d_model` 和 `d_ff`。
4. 输出形状保持 `(B, T, d_model)`。

## 参考实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.linear1(x)           # (B, T, d_ff)
        x = F.relu(x)                 # 激活
        x = self.linear2(x)           # (B, T, d_model)
        return x
```
