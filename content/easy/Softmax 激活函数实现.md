# 题目：自行实现一个简单的 softmax 激活函数

> 目标: 自行实现一个简单的 softmax，对输入矩阵的最后一个维度应用 Softmax，将原始的得分转换为概率分布。

## 数学定义

给定一个矩阵 $x \in \mathbb{R}^{B \times C}$, 其中 $B$ 是批次大小（样本数）, $C$ 是类别维度表示类别数，则 Softmax 对每个样本有如下定义:

$$
\text{Softmax}(x_{i,:}) = \frac{e^{x_{i,:}}}{\sum^C_{j=1}e^{x_{i,j}}}, \quad i=1,2,\dots,B
$$

其中, $x_{i,:}$ 表示第 $i$ 个样本的得分向量, $\sum^C_{j=1}e^{x_{i,j}}$ 表示对第 $i$ 个样本所有类别得分的指数值进行求和，确保输出的概率和为 1。

## 实现要求

- 对输入张量的每一行（即每个样本）应用 Softmax 函数。
- 用 PyTorch 实现，不能直接使用 `torch.nn.Softmax` 或 `F.softmax` 等高阶函数。
- 指数操作的数值稳定性保证

## 参考实现

```py
import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: (B, C)
        """
        max_x = torch.max(x, dim=-1, keepdim=True)[0]  # 对每行的最大值取出
        exp_x = torch.exp(x - max_x)   # 对每个元素减去最大值，避免溢出
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)  # 对每行进行求和
        return exp_x / sum_exp_x  # 对每个样本进行归一化，确保每行的和为 1
```
