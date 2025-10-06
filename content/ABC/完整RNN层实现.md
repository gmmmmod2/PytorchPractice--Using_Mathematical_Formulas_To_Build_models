# 题目：实现 RNN 层（序列展开，支持变长序列 mask，batch_first=True）

> 目标：在 单步 RNNCell 的基础上, 实现整段序列前向传播,逐步更新隐藏态并输出所有时间步隐藏序列。

## 数学定义

设输入为 $X\in \mathbb{R}^{B\times L\times d_{in}}$, 初始状态 $h_0\in\mathbb{R}^{B\times d_h}$, 时间步为 $t = 1,\dots, L$, 则更新公式为:

$$
h_t = \text{tanh}(W_hx_t+U_hh_{t-1}+b_h)
$$

对于掩码规则, 设 $m\in \{0,1\}^{B\times 1}$, 则公式表达为:

$$
h_t \leftarrow m_t \odot h_t + (1-m_t) \odot h_{t-1}
$$

逐步“冻结”填充位置, 保证无效位置的状态不被后续无意义步覆盖。

## 额外的输入/输出规定

- 输入 `x: (B, L, d_in)`
- 可选的输入 `h_0: (B, d_h)`, `mask: (B, L)`
- 输出: `h_all: (B, L, d_h)`, `h_last: (B, d_h)`

## 实现要求

- `Batch_first = True` 为固定要求
- 如没有提供 `mask` 则等价于 `mask` 全为 1
- `h_last` 的计算为, 对每个样本取最后一个 `mask = 1`的时间步隐藏状态；若整条为 0，则返回 `h_0` 或 零向量
- 仅前向，不含反传。

## 参考实现

```python
import torch
import torch.nn as nn

class MyRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCellRaw(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor=None, mask: torch.Tensor=None):
        """
        x:    (B, L, d_in)   batch_first
        h0:   (B, d_h) or None
        mask: (B, L) 0/1 or bool; 1=有效, 0=padding
        return: h_all(B,L,d_h), h_last(B,d_h)
        """
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
            assert mask.shape[:2] == (B, L)
            if mask.dtype != torch.float32:
                mask = mask.float()

        h_all = []
        for t in range(L):
            x_t = x[:, t, :]                        # (B, d_in)
            h_t = self.cell(x_t, h_prev)            # (B, d_h)
            m_t = mask[:, t].unsqueeze(-1)          # (B, 1)
            # 冻结无效步
            h_t = m_t * h_t + (1.0 - m_t) * h_prev
            h_all.append(h_t)
            h_prev = h_t

        h_all = torch.stack(h_all, dim=1)           # (B, L, d_h)

        # 计算每个样本“最后有效步”的索引
        # idx = 最大的 t 使得 mask=1；若全 0，用 -1 表示“无效”并回退到 h0/0
        with torch.no_grad():
            # 将无效置为 -inf，有效置为它的时间索引
            time_idx = torch.arange(L, device=x.device).view(1, L).expand(B, L).float()
            score = time_idx * (mask > 0.5).float() + (mask <= 0.5).float() * (-1e9)
            last_pos = torch.argmax(score, dim=1)   # (B,)
            has_valid = (score.max(dim=1).values > -1e8)  # (B,)

        # gather last hidden
        flat = h_all.reshape(B * L, d_h)
        offset = torch.arange(B, device=x.device) * L + last_pos
        h_last_from_seq = flat.index_select(0, offset.clamp(min=0)).reshape(B, d_h)

        # 如果没有有效步，用 h0（若无 h0 则 0）
        if h0 is None:
            h0_like = x.new_zeros(B, d_h)
        else:
            h0_like = h0
        h_last = torch.where(has_valid.unsqueeze(1), h_last_from_seq, h0_like)
        return h_all, h_last
```
