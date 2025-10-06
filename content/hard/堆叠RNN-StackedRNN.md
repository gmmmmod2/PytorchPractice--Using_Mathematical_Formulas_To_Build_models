# 题目：实现堆叠 RNN（Stacked RNN，含 LayerNorm + Dropout，支持 mask）

> 目标：在单向 RNN 层的基础上，实现 **N 层堆叠**：每层在时间展开后做 LayerNorm，层间加 Dropout。

## 数学定义

对第 $\ell$ 层（$\ell=1,\dots,N$）：

$$
H^{(\ell)} = \mathrm{RNN}_\ell(H^{(\ell-1)}),\quad
\widehat{H}^{(\ell)} = \mathrm{LayerNorm}(H^{(\ell)}),\quad
H^{(\ell)}_{\text{out}} =
\begin{cases}
\mathrm{Dropout}(\widehat{H}^{(\ell)}), & \ell < N\\
\widehat{H}^{(\ell)}, & \ell = N
\end{cases}
$$

mask 规则：与单层一致。

## 输入/输出

- 输入 `x: (B,L,d_in)`, `mask: (B,L)`, `num_layers ≥ 1`
- 输出 `h_all_top: (B,L,d_h) 顶层输出`, `h_last_top: (B,d_h) 顶层最后有效步`
- 可选 `all_layer_outputs`，每层输出列表

## 实现要求

- 固定 `batch_first=True`
- 各层 hidden_size 相同
- LayerNorm 针对最后一维
- Dropout 仅层间使用
- 仅前向

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
        return torch.tanh(x_t @ self.W_h.T + h_prev @ self.U_h.T + self.b_h)

class MyRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCellRaw(input_size, hidden_size)
    @property
    def hidden_size(self):
        return self.cell.hidden_size
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

        # last valid per sequence
        time_idx = torch.arange(L, device=x.device).view(1, L).expand(B, L).float()
        score = time_idx * (mask > 0.5).float() + (mask <= 0.5).float() * (-1e9)
        last_pos = torch.argmax(score, dim=1)
        has_valid = (score.max(dim=1).values > -1e8)
        flat = h_all.reshape(B * L, d_h)
        offset = torch.arange(B, device=x.device) * L + last_pos
        h_last_from_seq = flat.index_select(0, offset.clamp(min=0)).reshape(B, d_h)
        h0_like = x.new_zeros(B, d_h) if h0 is None else h0
        h_last = torch.where(has_valid.unsqueeze(1), h_last_from_seq, h0_like)
        return h_all, h_last

class StackedRNN(nn.Module):
    """
    多层堆叠 RNN：
      - 每层: 单向 RNN 展开 + LayerNorm
      - 层间: Dropout
      - 支持 mask / batch_first
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.0, use_layernorm: bool = True):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_ln = use_layernorm
        self.dropout_p = float(dropout)

        layers = []
        lns = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(MyRNNLayer(in_dim, hidden_size))
            lns.append(nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity())
            in_dim = hidden_size
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(lns)
        self.dropout = nn.Dropout(self.dropout_p) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, h0_list=None,
                return_all_layers: bool=False):
        """
        x: (B,L,d_in)
        mask: (B,L) 1=有效
        h0_list: None 或 长度=num_layers 的列表，每个 (B,d_h)
        return:
          h_all_top: (B,L,d_h)
          h_last_top: (B,d_h)
          (可选) all_layer_outputs: list[(B,L,d_h)]
        """
        B, L, _ = x.shape
        if h0_list is not None:
            assert len(h0_list) == self.num_layers

        inputs = x
        all_layer_outputs = []
        h_last_top = None

        for li in range(self.num_layers):
            h0 = None if h0_list is None else h0_list[li]
            h_all, h_last = self.layers[li](inputs, h0=h0, mask=mask)  # (B,L,d_h)
            # LayerNorm: 对最后一维（特征维）归一化
            h_all = self.layer_norms[li](h_all)
            # 层间 Dropout（顶层不再额外 dropout 也可以，但这里统一对“传给下一层”的张量做）
            if li < self.num_layers - 1:
                h_all = self.dropout(h_all)

            all_layer_outputs.append(h_all)
            inputs = h_all
            h_last_top = h_last  # 顶层的读出会不断覆盖，最终为最顶层

        h_all_top = inputs  # (B,L,d_h)

        if return_all_layers:
            return h_all_top, h_last_top, all_layer_outputs
        else:
            return h_all_top, h_last_top

```
