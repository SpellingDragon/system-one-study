# 从软件工程走向模型与算子：学习路径

读者只需具备软件工程本科基础：会写程序，理解函数、数组、测试和基本代数。这里不假设你学过深度学习、GPU 或昇腾。Python 例子中的 `@` 是矩阵乘法，`.shape` 是各维长度，`.T` 是转置；遇到新概念会先解释再使用。

这不是已经成功的产品宣传。System-One 同时保存了可用实现、负结果、研究候选和未完成任务。学习重点是：一个结果究竟证明了什么，怎么知道解释错了。

## 按前置知识阅读

| 顺序 | 内容 | 读完能做什么 | 状态 |
|---|---|---|---|
| 1 | [项目地图](01-project-map.md) | 分清数据、模型、决策、内核、评测和阶段目标 | 已产出 |
| 2 | [张量与梯度](02-tensors-and-gradients.md) | 手算一个线性层，理解 shape/stride 和链式法则 | 已产出 |
| 3 | [P1：从零训练与读出](03-p1-training-and-decisions.md) | 解释 BPE、attention、SFT、温度校准 | 已产出 |
| 4 | [P2：教师与训练配方](04-p2-teachers-and-training.md) | 分清 soft target、KL 方向、LoRA、on-policy、RL | 已产出 |
| 5 | [实验证据](05-experimental-evidence.md) | 识别数据泄漏、小样本、假绿和不公平性能比较 | 已产出 |
| 6 | [加速器与算子](06-accelerators-and-kernels.md) | 理解编译、内存层级、tiling、同步与 launch | 已产出 |
| 7 | [GDN 反向专题](../teaching_gdn_bwd_verification.md) | 沿真实问题练习隔离变量、参考验证和因果判断 | 已勘误接入 |

依赖关系是 `1 → 2 → 3 → 4`，`2+3 → 5`，`2+5 → 6 → 7`。可以先读项目地图和实验方法，再深入数学；不建议直接从 GDN 六梯度公式开始。

## 实验分三类

- **教学演示**：人为构造小数组说明一个概念，不声称复现历史故障。
- **历史证据阅读**：从已有日志核查输入、版本和结果，不需要重新租卡。
- **真实设备复验**：需要指定环境、原始实现与目标输入；CPU 演示不能代替它。

代码块若标为 Python 实验，可在已有 CPU Python 环境运行；使用 NumPy/PyTorch 的章节会写明依赖。复跑命令：

```bash
release/.venv/bin/python -P docs/learning_path/verify_examples.py            # 全部章节，各重复 10 次
release/.venv/bin/python -P docs/learning_path/verify_examples.py 02-tensors-and-gradients.md
```

不要为了读教材运行训练脚本、下载大权重或启用云端实例。实际验证记录见 [证据索引](../reassessment/evidence.md)。

## 共用记号

| 记号 | 含义 |
|---|---|
| B、H、T | batch 数、head 数、序列长度；状态矩阵另写 S，避免与 head 数 H 混淆 |
| C、DK、DV | 每个 chunk 的 token 数、key/query 宽度、value 宽度 |
| x、W、y | 输入、权重、输出；具体维度在公式旁给出 |
| L、∂L/∂x | 损失、损失对 x 的梯度 |
| p、q、τ | 概率分布、另一概率分布、温度；τ 不与序列长度 T 混用 |
| reference、actual | 参考实现结果、待测实现结果；golden 是参考实现的常用名称，不意味着绝不会错 |

每章自测题后都有解释性答案。能说清反例比背下规则更重要；例如“测试绿了”之后，继续问：测的是哪条路径、什么输入、哪个版本？
