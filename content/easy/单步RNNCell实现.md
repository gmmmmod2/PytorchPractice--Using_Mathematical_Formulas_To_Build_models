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
- 参数形状 `W_h: (d_h, d_in)`, `U_t: (d_h, d_h) `, `b_h: (d_h)`

## 实现要求

- 显示注册三个参数
- 形状断言与友好报错

## 参考实现

```python
import torch
import torch.nn as nn

class RNNCellRaw(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_h = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_h)
        nn.init.orthogonal_(self.U_h)
        nn.init.zeros_(self.b_h)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        B, d_in = x_t.shape
        _, d_h = h_prev.shape
        assert d_in == self.input_size and d_h == self.hidden_size, \
            f"shape mismatch: got x={x_t.shape}, h={h_prev.shape}, " \
            f"expect (*,{self.input_size}) and (*,{self.hidden_size})"

        h_t = torch.tanh(x_t @ self.W_h.T + h_prev @ self.U_h.T + self.b_h)
        return h_t
```
