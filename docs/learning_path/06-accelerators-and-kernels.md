# 6. 加速器、分块与 GDN：为什么能算对仍然很难

前置：[张量与梯度](02-tensors-and-gradients.md)、[实验证据](05-experimental-evidence.md)。本章不给跨版本可直接复制的 TileLang API 模板；代码是 CPU 数学演示。

## 6.1 CPU、GPU、NPU 的共同点与差异

三者都计算程序，但资源组织不同。CPU 擅长复杂控制流；GPU 通过大量并行工作隐藏延迟；NPU 为矩阵和向量计算提供专用执行资源。它们都受内存、数据搬运、同步和数值精度限制。

Ascend 涉及 Cube 矩阵计算、Vector 向量计算、标量控制与数据搬运通路。具体分核、指令和容量随型号、SDK 及后端变化；不能把 CUDA 的 thread/warp 代码直接按名字翻译，更不能把 950 的 API 假定为 910B 支持。[G05]

## 6.2 一段 DSL 到设备执行的过程

```text
Python DSL → IR（中间表示）→ 编译变换 passes → 目标代码
→ SDK 编译器／链接 → 可加载制品 → runtime 提交 → 设备执行 → 同步读取
```

IR 像编译器内部的数据结构，pass 改写它，例如安排内存或同步。代码生成成功只说明生成器接受输入；SDK 编译成功也不证明运行时地址和同步正确。

本项目的本地交叉编译减少了租卡等待：Mac 上的合适容器可以生成另一环境的目标制品。但它不能执行出 NPU 数值真值。编译绿、CPU 数学绿、NPU 数值绿是三份证据，不是一份。[G05、A09]

另外，Python 从哪里 import 包也影响结论。本轮出现当前目录的 tilelang 源码遮蔽 venv 包，导致找不到构建库；显式导入路径与 `-P` 后组件测试通过。这是环境选择问题，不能直接归罪数学实现。[V02]

## 6.3 内存层级：容量和通路都要对

```text
GM：设备全局内存，大容量
 ├─ 搬入 L1 → L0A/L0B → Cube 乘加 → L0C 累加结果
 └─ 搬入 UB → Vector 运算
结果通过该型号支持的搬运路径回到 UB、L1 或 GM
```

这是角色图，不是完整合法指令表。L1/UB/L0 的布局、对齐、读写粒度和同步条件由对应 SDK 路径决定。GM 的矩阵“看起来是二维连续数组”，搬入片上后可能采用分形或补齐布局。

一个 `[64,128]` fp16 tile 的裸数据占 `64×128×2=16384` 字节，即 16KiB。但实际容量还要加多个操作数、临时结果、padding、流水线多份 buffer 和编译器保留空间。只算一个数组不能证明整个 kernel 放得下。

## 6.4 tiling：分块不改变数学，却改变执行路径

矩阵乘的收缩维可以分段：`A@B = Σ_j A[:,j段]@B[j段,:]`。每段算部分和，第一段初始化，后面累加。BK 是块宽，`ceil(DK/BK)` 是循环次数。

### 实验 A：测尺寸不如测路径

依赖 NumPy。这是正确分块的数学对照，不复现任何后端缺陷。

```python
import numpy as np

A = np.arange(24, dtype=np.float64).reshape(4, 6) / 10
B = np.arange(18, dtype=np.float64).reshape(6, 3) / 7
expected = A @ B
for bk in (2, 4, 6):
    out = np.zeros((4, 3))
    iterations = 0
    for start in range(0, 6, bk):
        end = min(start + bk, 6)
        out += A[:, start:end] @ B[start:end, :]
        iterations += 1
    assert np.allclose(out, expected, rtol=1e-12, atol=1e-12)
    assert iterations == (6 + bk - 1) // bk
print('单块、多块、尾块数学对照通过')
```

BK=6 只有一轮，无法覆盖第二轮的地址偏移、累加和 buffer 复用；BK=4 还有尾块。若只测整除单块，就不应承诺多块尾块正确。

GDN 的 BK=DK 绕行只是不再走有问题的多块配置。当前直接 GM→L1 候选改法仍需真机对照。不能把教学例子中的“忘记加第二块”当成真实编译器机理。[A02]

## 6.5 同步：计算顺序和数据可见顺序

考虑两条异步执行通路：搬运器把 K 写入 buffer，矩阵单元读取 K。Python 源码先写 copy 再写 matmul，不自动证明硬件读取时搬运已完成。下一轮复用 buffer，还要等上一轮读者结束。

至少区分：生产者完成写入、消费者开始读取、最后一个消费者结束、下一次覆盖。这与软件工程的 happens-before 相似，但执行单元和存储空间更多。

`fill(0)` 是写数据，不等于建立所有依赖；自动同步意味着编译器推导这些依赖，不意味着任意布局和不合法访问都能被修好。手写 barrier 也不能修复错误地址或 dtype。[A02]

## 6.6 kernel 定义数、调用数与设备 launch

一个定义可以按不同形状实例化，多次调用；一次框架调用内部又可能发多个设备任务。程序还可能夹杂 ATen（PyTorch 底层算子）和 host 拷贝。

因此“20 个 kernel”和“每 chunk 约几十次调用”不是同一个数；“≤2 kernel”若不说明是否包括前置和 host 运算，无法作为完整系统预算。v3 当前可按源码数 TL callable，但物理 launch 数需要设备 trace。[A06]

把多个操作融合可能减少提交和搬运，也可能增加片上容量压力、减少并行度或改变舍入次序。先有容量账和受控原型，再量速度；不能拿前向的耗时当作反向融合后保证值。

## 6.7 GDN 先从一个状态矩阵理解

标准 attention 可以保留大量历史 K/V。Gated DeltaNet 用状态矩阵 S 概括历史信息。以单 head 的项目参考约定为例，`S:[DV,DK]`，q/k 为 DK 维、v 为 DV 维。

```text
α_t = exp(g_t)                         # 衰减系数
S_decay = α_t * S_prev
prediction = S_decay @ k_t             # 当前 key 下的旧预测
u_t = β_t * (v_t - prediction)          # 用新信息修正
S_t = S_decay + outer(u_t, k_t)          # outer 是外积，结果 [DV,DK]
o_t = S_t @ q_t                        # 输出 [DV]
```

这里 g 是 log 衰减量，不能把 g 本身当 α。β 控制更新强度，q 用于读出，k 用于写入／匹配。状态压缩可以节省历史存储，但不是无损保存所有历史 token。[A09]

### 实验 B：两步状态更新

```python
import numpy as np

state = np.zeros((2, 2))
keys = np.eye(2)
values = np.array([[2., 1.], [4., 2.]])
outputs = []
for k, v in zip(keys, values):
    decayed = 0.5 * state
    update = v - decayed @ k             # β=1
    state = decayed + np.outer(update, k)
    outputs.append(state @ k)            # 此例 q=k
assert np.array_equal(state, np.array([[1., 4.], [.5, 2.]]))
assert np.array_equal(outputs, values)
print('旧列衰减，新列写入')
```

第二步后第一列保留一半，第二列写入新信息。改变查询 q，就会读出不同的历史组合；输出等于 values 只是本例正交 key、β=1 的特殊结果，不是 GDN 恒等式。

## 6.8 为什么反向要从后往前

S_t 同时影响当前输出和未来状态。损失对 S_t 的梯度要汇总这两条路径，再传给 S_(t−1)。在完整数学中可以写成 Jacobian（多输入多输出的导数矩阵）链式法则；实现常把它整理成矩阵乘和逐元素运算。

chunked 形式把连续 C 个 token 的更新代数重组，生成 A、W、Kt、Vnew 等中间量。跨块矩阵伴随状态 Δ' 沿反方向传递；dk 是每个输入 key 的最终梯度，不作为下一块状态；dg 另有累计和导数带来的 suffix/carry。[A01]

接下来阅读 [GDN 反向验收专题](../teaching_gdn_bwd_verification.md)：从公式、参考和三种误差尺子，追到真实多块、切片和探针问题。

## 6.9 教学案例：支持面裁定的推理（真实决策，2026-10-24）

本仓当天有一个真实裁定：GDN 反向要支持哪些输入？三个候选维度与结论：

| 维度 | 事实 | 裁定 |
|---|---|---|
| fp32 档 | 从未验收；bf16 已被在案裁决排除；训练主档 fp16 | 显式降级声明 |
| 尾块（T 不是 C 的倍数） | 当前实现直接拒绝；真实训练的变长序列必然遇到 | host 侧补齐到整倍数 + 验证补齐不改变结果 |
| DK≠DV | 从未支持；目标模型 GDN 头 16 V + 16 QK、head 维 128 | 降级声明（结构上 DK=DV） |

推理要点：**降级不是偷懒的代名词**。DK≠DV 的降级依据是目标模型的公开结构（领域事实），不是“没做”；而尾块不能降，因为变长序列是训练的真实输入。同一个“不做”，一个有据、一个不可。这是《行事风格》中“边界画在语义单元上”的具体样子。

host 补齐为什么可能可行？补一个“哑 token”：q 任意、k=0、v=0、g=0、β=0。代入 6.7 的状态更新，三个零各司其职：k=0 使外积不写入，β=0 使修正量为零，g=0 使旧状态不衰减。

### 实验 C：哑 token 不改变状态

依赖 NumPy；只验证状态传递这一步，不是完整证明。

```python
import numpy as np

state = np.array([[1., 2.], [3., 4.]])
k_pad, v_pad, g_pad, beta_pad = np.zeros(2), np.zeros(2), 0.0, 0.0
decayed = np.exp(g_pad) * state
update = beta_pad * (v_pad - decayed @ k_pad)
new_state = decayed + np.outer(update, k_pad)
assert np.array_equal(new_state, state)
assert not np.array_equal(np.exp(-0.5) * state, state)   # 若误填 g_pad≠0，旧状态被错误衰减
print('哑 token 状态不变；g 的填充值必须是 0')
```

漏掉任何一个零（尤其 g）补齐就会改变结果——所以“pad 到整倍数”不是随手 torch.cat 一推零，而是一份每个字段都有理由的契约。

这个实验**不构成完整验证**：反向链还有 dP、gam2 等块内量，dO 的补齐值与 mask 的消费方式都可能引入差异。工程上的完整验收是：对整条实现（前向+反向）做“补齐前后对有效位置的输出与梯度逐项对照”（即裁定中“golden 同 pad 对照”的含义）。单步演示指出要验什么，完整对照证明它成立——两者缺一不可。裁定原文见 [工程评估 §5](../reassessment/05-plan-recommendations.md)。

## 6.10 自测

1. 编译器产出了可执行制品，能否省去设备数值测试？——不能，合法编译不证明运行地址、同步和舍入正确。
2. 一轮循环成功，是否覆盖 buffer 下一次复用？——没有，需要多轮与边界输入。
3. 增加 barrier 能修 dtype 不匹配吗？——不能，类型与同步是不同契约。
4. 为什么 Δ' 需要跨块而 dk 不需要？——Δ' 表示未来损失对状态的影响，dk 是某个输入 key 的最终导数；看读写消费者，而非名字猜递推。
5. 一核版本一定快吗？——不一定，资源压力和并行度可能抵消融合收益，需同工作量 A/B。

[学习导航](README.md) · [工程专题](../reassessment/04-ascend-gdn.md)
