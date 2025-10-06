# 题目：实现双向 RNN 层（支持变长序列 mask，`batch_first=True`）

> 目标：在单向 RNN 的基础上，实现**前向 + 反向**两个方向的序列展开，并在时间维拼接得到双向隐藏序列。

## 数学定义

设输入 $X\in\mathbb{R}^{B\times L\times d\_{in}}$、mask $M\in\{0,1\}^{B\times L}$。

- **正向链**（$t=1\to L$）：

$$
\overrightarrow{h}_t=\tanh(W_f x_t + U_f \overrightarrow{h}_{t-1}+b*f),\quad
\overrightarrow{h}\_t \leftarrow m_t\overrightarrow{h}\_t+(1-m_t)\overrightarrow{h}*{t-1}
$$

- **反向链**（$t=L\to 1$）：

$$
\overleftarrow{h}_t=\tanh(W_b x_t + U_b \overleftarrow{h}_{t+1}+b*b),\quad
\overleftarrow{h}\_t \leftarrow m_t\overleftarrow{h}\_t+(1-m_t)\overleftarrow{h}*{t+1}
$$

- **输出拼接**：

$$
H_t=\big[\,\overrightarrow{h}_t\ ;\ \overleftarrow{h}_t\,\big]\in\mathbb{R}^{2d_h}
$$

读出（可选）：

- `h_last_fwd`：最后有效步的正向隐藏态
- `h_last_bwd`：第一个有效步的反向隐藏态

## 输入/输出

- 输入 `x: (B,L,d_in)`，
- 可选输入 `mask: (B,L)`, `h0_fwd/h0_bwd: (B,d_h)`
- 输出 `h_bi: (B,L,2d_h)`, `h_last_fwd: (B,d_h)`, `h_last_bwd: (B,d_h)`

## 实现要求

- 固定 `batch_first=True`
- 若未提供 `mask`，等价全 1
- 反向链：对输入序列和 mask 做反转，再展开，最后反转回去
- 仅前向

## 参考实现

```python
import torch
import torch.nn as nn

# 也可从你现有文件导入
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
        return torch.tanh(x_t @ self.W_h.T + h_prev @ self.U_h.T + self.b_h)

class MyRNNLayer(nn.Module):
    """单向 RNN 层：展开序列 + mask 冻结 + h_last 读出。"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCellRaw(input_size, hidden_size)
    def forward(self, x: torch.Tensor, h0: torch.Tensor=None, mask: torch.Tensor=None):
        B, L, d_in = x.shape
        d_h = self.cell.hidden_size
        if h0 is None:
            h_prev = x.new_zeros(B, d_h)
        else:
            assert h0.shape == (B, d_h)
            h_prev = h0
        if mask is None:
            mask = x.new_ones(B, L, dtype=torch.float32)
        else:
            if mask.dtype != torch.float32:
                mask = mask.float()

        h_all = []
        for t in range(L):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_prev)
            m_t = mask[:, t].unsqueeze(-1)
            h_t = m_t * h_t + (1 - m_t) * h_prev
            h_all.append(h_t)
            h_prev = h_t
        h_all = torch.stack(h_all, dim=1)  # (B,L,d_h)

        # last valid readout
        time_idx = torch.arange(L, device=x.device).view(1, L).expand(B, L).float()
        score = time_idx * (mask > 0.5).float() + (mask <= 0.5).float() * (-1e9)
        last_pos = torch.argmax(score, dim=1)                # (B,)
        has_valid = (score.max(dim=1).values > -1e8)         # (B,)
        flat = h_all.reshape(B * L, d_h)
        offset = torch.arange(B, device=x.device) * L + last_pos
        h_last_from_seq = flat.index_select(0, offset.clamp(min=0)).reshape(B, d_h)
        h0_like = x.new_zeros(B, d_h) if h0 is None else h0
        h_last = torch.where(has_valid.unsqueeze(1), h_last_from_seq, h0_like)
        return h_all, h_last

class BiRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fwd = MyRNNLayer(input_size, hidden_size)
        self.bwd = MyRNNLayer(input_size, hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None,
                h0_fwd: torch.Tensor=None, h0_bwd: torch.Tensor=None):
        """
        x: (B,L,d_in); mask:(B,L) 1=有效
        return:
          h_bi: (B,L,2*d_h)
          h_last_fwd: (B,d_h)
          h_last_bwd: (B,d_h)  # 原时间轴上的“第一个有效步”的反向隐藏态
        """
        B, L, _ = x.shape
        # forward chain
        h_fwd, h_last_fwd = self.fwd(x, h0=h0_fwd, mask=mask)
        # backward chain: reverse time
        x_rev = torch.flip(x, dims=[1])
        mask_rev = None if mask is None else torch.flip(mask, dims=[1])
        h_bwd_rev, h_last_bwd_rev = self.bwd(x_rev, h0=h0_bwd, mask=mask_rev)
        h_bwd = torch.flip(h_bwd_rev, dims=[1])

        # 在原时间轴上，“第一个有效步”的反向读出 = 反转序列上的 last_valid
        h_last_bwd = h_last_bwd_rev

        h_bi = torch.cat([h_fwd, h_bwd], dim=-1)  # (B,L,2*d_h)
        return h_bi, h_last_fwd, h_last_bwd
```
