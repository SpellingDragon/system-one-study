# 4. P2：教师、蒸馏与训练配方

前置：[P1 训练与决策](03-p1-training-and-decisions.md)。下面从损失和数据流理解方法，不靠函数名判断算法。

## 4.1 教师为什么不只给一个答案

硬标签 `[0,1,0]` 表示第二项正确。soft target（软目标）如 `[0.1,0.7,0.2]` 还表达候选间的相对倾向。教师是产生目标分布的模型，学生学习接近这个分布。

教师可以错；大模型也可能因任务、模板或格式不匹配而产生差伪标。随机初始化的小模型可用于检查“缓存→装配→loss”的管道，不能作为有知识教师的替代证据。System-One 的 `#scaffold-cpu` 后缀就是区分身份的记号。[P201–P203]

缓存要同时对应模型版本、输入和候选语义。命中只说明找到了一个键，不说明教师正确，也不说明该条样本属于训练集。

## 4.2 soft-CE 与 KL：把方向写出来

令 t 是教师概率，s 是学生概率，二者在同一候选集合上归一化。

```text
soft-CE(t,s) = -Σ t_i log(s_i)
KL(t||s)    =  Σ t_i log(t_i/s_i) = soft-CE(t,s) - H(t)
H(t)        = -Σ t_i log(t_i)
KL(s||t)    =  Σ s_i log(s_i/t_i)
```

教师固定时 H(t) 是常数，最小化 soft-CE 与 KL(t||s) 对学生的梯度一致。交换 KL 两边会改变权重和梯度；“forward/reverse”要先指明谁是目标、谁是近似，最稳妥的写法是完整公式。

### 实验 A：名字相似，数值不同

标准库实验，只用严格正概率，避开零概率的特殊定义。

```python
import math

t = [0.9, 0.1]
s = [0.5, 0.5]
def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q))

forward = kl(t, s)
reverse = kl(s, t)
ce = -sum(a * math.log(b) for a, b in zip(t, s))
entropy = -sum(a * math.log(a) for a in t)
assert abs(forward - (ce - entropy)) < 1e-12
assert abs(forward - reverse) > 0.1
print(f'KL(t||s)={forward:.6f}, KL(s||t)={reverse:.6f}')
```

本项目名为 `reverse_kl` 的函数正常域实际计算 KL(t||s)。这可以是合理的损失，但若计划承诺另一个方向，就需要修订算法契约或实现，不能靠单测把名字“证明正确”。[P204]

[PyTorch KL 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html)中 input 是 log 概率、target 默认是概率；参数位置不等于数学上 KL 的书写顺序。按 batch 求平均也不同于按所有元素平均。

## 4.3 per-token、选项级与 on-policy 是三条轴

- **选项级**：一次对 A/B/C 等候选计算分布差异。
- **per-token**：在生成序列的每个位置、对应前缀条件下比较词表分布。
- **on-policy**：数据生成与当前学生策略相联系；在序列蒸馏里，通常由学生生成序列，教师评价这些前缀上的行为。

[GKD 原文](https://arxiv.org/abs/2306.13649v3)针对学生训练时看到的序列与推理时自己生成的序列不一致的问题。它允许不同损失，并不等于“某个固定 KL 方向”。

按一次分数取 top-k 选项不等于随机生成序列；在训练前选好子集后重复更新，也不等于持续跟随变化后的学生分布。本项目当前实现正需要厘清这几层含义。[P204]

### 实验 B：旧候选集合会过时

这是过程演示，不是 GKD 算法实现。

```python
old_scores = [3.0, 2.0, 1.0]
new_scores = [1.0, 2.0, 3.0]
def top2(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:2]

cached = top2(old_scores)
current = top2(new_scores)
assert cached == [0, 1]
assert current == [2, 1]
assert cached != current
print('旧子集不代表当前策略')
```

## 4.4 LoRA：只训练一个低秩增量

对行向量 x，普通线性层 `y=xW.T`。LoRA 固定 W，训练较小矩阵 A、B：

```text
W:[d_out,d_in]；A:[r,d_in]；B:[d_out,r]
y = xW.T + (α/r) * (xA.T)B.T
```

r 小于两端维度时，可训练参数从 `d_out*d_in` 减为 `r*(d_in+d_out)`。它限制的是权重更新形态，不是免除主干计算。原始来源：[LoRA](https://arxiv.org/abs/2106.09685v2)。

### 实验 C：零初始化增量保持初始输出

依赖 NumPy；这里用 α/r=1。

```python
import numpy as np

x = np.ones((1, 8), dtype=np.float64)
w = np.arange(48, dtype=np.float64).reshape(6, 8) / 50
A = np.arange(16, dtype=np.float64).reshape(2, 8) / 20
B = np.zeros((6, 2), dtype=np.float64)
base = x @ w.T
out = base + (x @ A.T) @ B.T
assert np.array_equal(base, out)
assert A.size + B.size == 28 < w.size
B[0, 0] = 0.1
out2 = base + (x @ A.T) @ B.T
assert not np.array_equal(base, out2)
print('LoRA 零扰动与增量形态通过')
```

真实训练还涉及 dropout、scale、参数加载、merge、反向和精度；该例没有证明这些实现。System-One 的 kernel 分支只接 LoRA 增量，不自动替换整个 backbone。[P203]

## 4.5 梯度检查点为何省内存又增加计算

普通反向需要前向中间量。梯度检查点保留部分边界，反向时重跑一段前向来恢复其余中间量。它交换的是“激活存储”和“重算时间”，不是把模型权重变小。

随机操作的状态、重算路径和梯度连接需要一致。看到 `gradient_checkpointing_enable()` 被调用，只证明开关动作；峰值内存下降多少要在同输入、同模型、同设备下量。被冻结的主干可能仍要把梯度传给更早的 LoRA，见第 2 章。

## 4.6 RL 与 proper scoring rule

监督学习给目标答案；强化学习（RL）给行为反馈 reward，再让策略倾向更高预期回报的行为。生成什么、怎样评分、梯度如何估计，都是算法的一部分。

一个二元事件真实概率为 q，报告概率 p。对数评分的期望为 `q log p + (1-q) log(1-p)`；对 p 求导为 `q/p-(1-q)/(1-p)`，在 p=q 时达到最大。这说明在固定真实分布的条件下，诚实报告概率最有利，称为 proper scoring rule。

[Rewarding Doubt](https://arxiv.org/abs/2503.02623v6)研究生成答案时表达置信度。它与本项目的零生成选项概率不自动同构：动作空间、置信度如何表示、reward 依赖什么都可能不同。理论最优策略的性质不等于有限数据 RL 训练一定达到校准。

## 4.7 自测

1. 损失函数叫 reverse_kl，能据名称判断方向吗？——不能，写出公式并用不对称分布对照。
2. top-k 选项来自学生，是否就是序列 on-policy？——不是，概率空间和更新时间也要一致。
3. LoRA B=0 时是否两矩阵的梯度都必为零？——不是，通常 B 可收到梯度，A 的梯度受 B 影响；零初始增量不等于无法学习。
4. 教师前向数为 0 是好事还是坏事？——缓存复用时可能正确；第一次产标时若也为 0，就需核查是否真使用了教师。
5. 共享 log score 是否足以证明复现某论文？——不够，还需核对状态、动作、数据、优化器与实验协议。

[学习导航](README.md) · [P2 评估](../reassessment/03-p2-chain.md) · [证据索引](../reassessment/evidence.md)
