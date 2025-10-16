# 题目：实现 Vanilla RNNCell（单步，tanh）

> 目标：根据原始 RNN 公式实现单步 RNNCell 前向，支持批次维度。

## 数学定义

给定当前步的输入 $x_{t} \in \mathbb{R}^{d_{in}}$, 上一步的隐藏态 $h_{t-1}$, 更新方式则有如下:

$$
\tilde{h} = \text{tanh}(W_hx_t+U_hh_{t-1}+b_h),\quad h_t=\tilde{h}
$$

## 额外的输入/输出规定

- 对于输入 `x_t: (B, d_in)`, `h_p: (B, d_h)`
- 对于输出 `h_t: (B, d_h)`
- 参数形状 `W_h: (d_h, d_in)`, `U_h: (d_h, d_h) `, `b_h: (d_h)`

## 实现要求

- 显示注册三个参数

## 参考实现

```python
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, d_in: int, d_h: int):
        super().__init__()

        self.w_h = nn.Parameter(torch.empty(d_in, d_h))
        self.u_h = nn.Parameter(torch.empty(d_h, d_h))
        self.b_h = nn.Parameter(torch.empty(d_h))
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_h) # 选择均匀分布初始化
        nn.init.orthogonal_(self.u_h)     # 正交初始化
        nn.init.zeros_(self.b_h)          # 偏置项通常为 0

    def forward(self, x_t: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, d_in)
        h_p: (B, d_h)
        """
        h_t = self.tanh(x_t @ self.w_h + h_p @ self.u_h + self.b_h)
        return h_t
```
