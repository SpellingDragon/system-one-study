# 2. 张量与梯度：读懂模型代码前的最小数学

前置：[项目地图](01-project-map.md)。本章只用数组、乘加和导数。实验是 CPU 教学演示，不训练模型，也不下载数据。

## 2.1 tensor 是带形状的数组

标量只有一个数；向量是一列数；矩阵有行和列；tensor（张量）可以有更多维。`shape=(2,3)` 表示 2 行、每行 3 个元素，不表示这些数字的业务含义。

模型常用 `[B,H,T,D]`：B 个样本，H 个 head，每个 head 有 T 个位置、每个位置 D 个特征。head 是把特征分组计算的方式，不是又多了一份数据标签。

阅读一个运算时先写形状。例如 `Q:[T,DK]`、`K:[T,DK]`，则 `Q @ K.T:[T,T]`。收缩的是 DK：每个结果元素等于两条 DK 长向量的逐项乘积之和。

```text
A:[M,K]  ×  B:[K,N]  →  C:[M,N]
C[i,j] = Σ(k=0..K-1) A[i,k] * B[k,j]
```

`A * B` 通常是逐元素乘法，`A @ B` 是矩阵乘法。把两者弄混，即使 shape 恰好相同也会得到不同的数学。

## 2.2 shape 相同，内存排列不一定相同

连续数组按某个顺序紧挨存放；stride（步长）告诉你每个维度索引增加 1 时跳过多少元素。按元素计算的地址为：

```text
offset = storage_offset + i0*stride0 + i1*stride1 + ...
```

视图可以共享原内存，只改变 shape、stride 或起点。切中间一段，不会自动把各个 head 的片段重新拼在一起。

### 实验 A：切片后的间隙

依赖 PyTorch；独立运行本代码块。

```python
import torch

x = torch.arange(24).reshape(1, 2, 4, 3)
s = x[:, :, :2, :]
assert x.is_contiguous()
assert s.shape == (1, 2, 2, 3)
assert s.stride() == (24, 12, 3, 1)
assert not s.is_contiguous()
c = s.contiguous()
assert torch.equal(c, s)
assert c.is_contiguous()
x1 = torch.arange(12).reshape(1, 1, 4, 3)
assert x1[:, :, :2, :].is_contiguous()
print('stride 演示通过')
```

B=1、H=1 时某些维度只有一个位置，无需跨越其 stride，所以小用例可能碰巧连续。真实 GDN 的 H>1 切片问题与此性质有关，但这个 CPU 演示并没有复现 NPU 编译器故障。[A01]

`.contiguous()` 可能分配和拷贝；这解决布局契约，不免费提高性能。是否需要它，由消费者接受的 stride 决定。

## 2.3 loss 把“哪里错了”压成一个数

设输入行向量 `x:[1,2]`、权重 `W:[2,2]`、输出 `y=xW:[1,2]`，目标为 t。用平方损失：

`L = 1/2 Σ_j (y_j-t_j)^2`。

取 `x=[1,2]`，W 两行为 `[1,2]`、`[3,4]`，则 y=`[7,10]`。目标 `[5,8]`，两个误差都为 2，L=4。

梯度表示局部敏感度：某个输入增加极小量，L 大约增加多少。它不是“这个输入本身多大”，也不是概率。

## 2.4 链式法则就是沿调用关系回传敏感度

若 `y=f(x)`、`L=g(y)`，标量链式法则为 `dL/dx=(dL/dy)(dy/dx)`。对矩阵，规则仍是这个，只是要正确累加共享路径：

- `dy = ∂L/∂y = y-t`，形状 `[1,2]`；这里 dy 是梯度简写，不是额外输入 y。
- `dW = x.T @ dy`，形状 `[2,2]`。
- `dx = dy @ W.T`，形状 `[1,2]`。

数值为 `dy=[2,2]`、`dW=[[2,2],[4,4]]`、`dx=[6,14]`。

### 实验 B：手算、autograd 与有限差分

autograd 会记录计算图并应用链式法则。有限差分则扰动一个参数，用函数值近似导数，两者是不同检查路径。

```python
import torch

x = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
w = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
t = torch.tensor([[5., 8.]], dtype=torch.float64)
y = x @ w
loss = ((y - t) ** 2).sum() / 2
loss.backward()
assert loss.item() == 4.0
assert torch.equal(x.grad, torch.tensor([[6., 14.]], dtype=torch.float64))
assert torch.equal(w.grad, torch.tensor([[2., 2.], [4., 4.]], dtype=torch.float64))
eps = 1e-6
plus, minus = w.detach().clone(), w.detach().clone()
plus[0, 0] += eps
minus[0, 0] -= eps
lp = ((x.detach() @ plus - t) ** 2).sum() / 2
lm = ((x.detach() @ minus - t) ** 2).sum() / 2
assert abs(((lp - lm) / (2 * eps)).item() - 2.0) < 1e-6
print('梯度三方核对通过')
```

有限差分不是任何情况下都准：eps 太大有截断误差，太小有浮点相消，不光滑点处也要特别处理。不能只选一个方便的 eps 就认定所有梯度正确。

## 2.5 冻结权重不等于不需要反向

设 `y=w*a`，w=2 固定，a=3 可训练，`L=y²`。虽然不更新 w，仍需经过乘 w 的反向才能得到 `dL/da=2*y*w=24`。

```python
import torch

a = torch.tensor(3., requires_grad=True)
w = torch.tensor(2.)
loss = (w * a) ** 2
loss.backward()
assert a.grad.item() == 24.0
assert w.grad is None
print('冻结参数仍传梯度')
```

这解释了 LoRA 训练为何可能需要冻结 backbone 中的梯度传播：如果可训练模块处于上游，梯度需要穿过后面的冻结运算。只有计算图上不需要求任何上游梯度的部分，才可整体去掉反向。

## 2.6 浮点数不是精确实数

fp16、bf16、fp32、fp64 分配给指数和有效数字的位数不同。fp16 的单位舍入约 `2^-11`，bf16 约 `2^-8`；bf16 的指数范围较大，但有效数字较少。它们都不是“误差永远等于某个固定常数”。

误差与数值大小、运算顺序、抵消、输入转换和累计路径有关。即使最后输出是 fp32，之前已丢掉的信息也不会自动恢复。GDN 中的 fp32 状态和 hi/lo 拆分就是针对不同误差路径的措施，不是把所有 dtype 改大就一定有效。[A01–A02]

## 2.7 自测

1. `[8,16] @ [16,4]` 的输出和收缩维是什么？
2. 为什么 H=1 测试不能替代 H=16 的切片检查？
3. 参数被冻结，是否总能跳过所在层 backward？
4. fp64 参考与被测实现共享同一个错误公式，二者一致说明什么？

答案：1. 输出 `[8,4]`，收缩维 16。2. 长度为 1 的维度不会实际跨 stride，可能隐藏间隙。3. 不能，上游可训练参数可能需要穿过它的梯度。4. 只说明两实现相符，不证明公式正确；还需解析小例子、独立推导或有限差分。

[学习导航](README.md) · [P1 评估](../reassessment/02-p1-chain.md) · [证据索引](../reassessment/evidence.md)
