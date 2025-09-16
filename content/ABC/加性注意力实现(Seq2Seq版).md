# 题目：加性注意力(Seq2Seq 对齐)

> 目标 实现加性注意力，用于对齐 decoder 当前隐状态 $s_t$ 与 encoder 序列 $H$。

## 数学定义

给定 $H=\{h_1,\dots,h_L\}, s_t$：

$$
e_{t,i} = v^\top \tanh(W_h h_i + W_s s_t),\quad
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})},\quad
c_t = \sum_i \alpha_{t,i} h_i
$$

## 额外的输入/输出规定

- `H` 形状 $(B,L,d_h)$，`s_t` 形状 $(B,d_s)$。
- `W_h` 和 `W_s` 都将 `H` 和 `s_t` 映射到维度 `d_attn`
- 允许 padding mask(形状 $(B,L)$，`0` 为 pad), 对无效位置加上一个足够大的负数。

## 参考实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, d_h: int, d_s: int, d_attn: int):
        super().__init__()
        self.W_h = nn.Parameter(torch.empty(d_attn, d_h))
        self.W_s = nn.Parameter(torch.empty(d_attn, d_s))
        self.v   = nn.Parameter(torch.empty(d_attn))

        self.act = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot 初始化：适合 tanh
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.W_s)
        # v 当作列向量，同样用 xavier（实现上需要 2D 张量）
        nn.init.xavier_uniform_(self.v.unsqueeze(0))  # (1, d_attn)

    def forward(self, H: torch.Tensor, s_t: torch.Tensor, padding_mask: torch.Tensor | None = None):
        Wh = H @ self.W_h.T                   # (B,L,d_attn)
        Ws = (s_t @ self.W_s.T).unsqueeze(1)  # (B,1,d_attn)
        e  = self.act(Wh + Ws)                # (B, L, d_attn)
        if padding_mask is not None:
            e = e.masked_fill(~padding_mask, float('-inf'))

        alpha = F.softmax(e, dim=-1)                    # (B, L)
        c = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) # (B, d_attn)
        return c, alpha
        
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size,d_h,d_s):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        self.W_s=nn.Linear(d_s,hidden_size,bias=False)
        self.W_h = nn.Linear(d_h, hidden_size,bias=False)
        self.v = nn.Linear(hidden_size, 1,bias=False)

    def forward(self,H,s_t,padding_mask=None):
        """
        :param H:编码器所有隐藏状态 [B,L,d_h]
        :param s_t:解码器当前隐藏状态 [B,d_s]
        :param padding_mask: [B,L]
        :return: context [B,hidden_size]
                 attn_weights [B,L]
        """
        B,L=H.shape[:2]
        s_t=s_t.unsqueeze(1).expand(-1,L,-1) #[B,L,d_s]
        H=self.W_h(H)
        s_t=self.W_s(s_t)
        score=self.v(torch.tanh(H+s_t)).squeeze(-1) #[B,L]
        if padding_mask is not None:
            score=score.masked_fill(padding_mask==0,float("-inf"))
        attn_weights=F.softmax(score,dim=-1)
        context=torch.bmm(attn_weights.unsqueeze(1),H).squeeze(1) #[B,hidden_size]
        return context,attn_weights
        
if __name__=='__main__':
    hidden_size=8
    d_h=4
    d_s=4
    B=2
    L=8
    H=torch.randn(B,L,d_h) #[2,8,4]
    s_t=torch.randn(B,d_s) #[2,4]
    Attention=BahdanauAttention(hidden_size,d_h,d_s)
    context,attn_weights=Attention(H,s_t,padding_mask=None)
    print(context.shape) #[2,8]
    print(attn_weights.shape) #[2,8]
```
