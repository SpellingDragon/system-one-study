# B3_GRADCHECK_DESIGN — 八门面「数值梯度判决件」设计（P1-1y：把悬空的 gradcheck NPU 档落地）

**交付件**：`release/ascend/gradcheck_card.py`（独立可跑，卡窗一条命令）。
**定位**：`p2-13 tasks.md` 条目 1 的「gradcheck NPU 档」此前**全库从未落地**（只有 host fp64
`torch.autograd.gradcheck`，`tests/test_ascend_train_step.py`）。C3 复核判其悬空；编排者裁决
**落地**为「第三窗可一条命令跑的判决件」，本文件即其设计说明书与自证留档。

**一句话**：对八个可微门面，在**同一次运行**里同时取「fp32 中心差分数值梯度」与「门面 autograd
解析梯度」，按 Frobenius 相对误差对拍；并且**只有在同一趟的三证（前向编译键 ∧ 无未申报 blocker ∧
无未申报回落）齐了**，这个对拍结果才算数——否则判 **INVALID**（不是 PASS，也不是 FAIL）。

---

## 0. 判决面与判据（复用，不另造）

| 项 | 取值 | 来源 |
|---|---|---|
| 判决面 | linear / add_ln / rope / attn_sw / gdn_delta / gdn_conv / readout / lora_apply | `ascend/autograd_asc.py` 八门面 |
| 对拍阈值 | `rtol=1e-4`, `atol=1e-5` | **沿用** `tests/test_ascend_gradcheck.py:49` 的 `TOL`（不另造） |
| 判决行 | `GRADCHECK-<op> PASS max_rel=<v> eps=<e>` / `FAIL …` / `INVALID reason=no-kernel\|blockers:<k>\|fallback` | 本件 |
| 末行 | `GRADCHECK-SUMMARY ops=8 pass= fail= invalid= kernel_used=<n>/<8>` | 本件 |
| 退出码 | pass 全绿 `0` / 有 FAIL `1` / 有 INVALID `2`（FAIL 优先于 INVALID） | 本件 |

对拍量的定义（Frobenius 相对误差，把沿各输入元素、各方向拼接后统一取范数）：

```
abs_err = || num_grad - ana_grad ||_F          # 各输入的梯度张量拼接后整体范数
ana_norm = || ana_grad ||_F
rel      = abs_err / max(ana_norm, 1e-30)      # 判决行里的 max_rel 字段
PASS  ⇔  abs_err <= atol + rtol * ana_norm     # 与 TOL 同数（1e-4 / 1e-5）
```

> **口径声明（与字面 `assert_close` 的差异）**：`tests/test_ascend_gradcheck.py` 用的是**逐元素**
> `torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)`。本件复用的是**同一组数**
> （rtol / atol），但作用是**张量范数级**不等式。原因见 §3.4：fp32 中心差分的**逐元素**噪声地板
> （对近零元素）会高于 `atol=1e-5`，逐元素口径在本件会**误报** `add_ln / gdn_conv / gdn_delta` 为
> FAIL——那是量具的精度极限，不是被测件的缺陷。范数级口径保留了「能不能咬住 ×0.1% 偏差」这一
> 全部判别力（§5.3：×1.001 负例 `rel≈1e-3`，是阈值的 10 倍），故如此收敛，并把差异如实登记。

---

## 1. 为什么用中心差分，而不是「解析对拍」

**做法**：解析侧 = 门面自身的 `autograd` 反传（走内核反向 / fp32 闭式 / cpu 尺子，即**生产那一路**）；
数值侧 = 只对**前向**做中心差分：`num[i][k] = [⟨f(x+εe_k), dy⟩ − ⟨f(x−εe_k), dy⟩] / (2ε)`。

**为什么数值侧必须是「只读前向的差分」而不是另写一份解析式/闭式**：

1. **免于「同源共错」**：另写一份解析参考，等于把「反向链式法则」这件要验的事，用同一套推导
   再写一遍——两份推导若犯同一个错（比如都漏掉 softmax 反向的 `Σdp·p` 项、或 LN 反向的两个均值），
   对拍会「一致地错」而放行。差分**只看前向**，不做任何链式推导，因此能独立咬住**任何**反向
   公式/接线错误（含 `Function` 参数顺序、`None` 占位、残差直传合并等）。
2. **免于「量到别处」**：本件的靶子是「门面 + 内核」这条**接线**的正确性，不是「数学公式重推一遍」。
   差分把被测物当成黑箱前向，正是要验的那条路。
3. **与既有 host 档同族**：`tests/test_ascend_train_step.py` 的 fp64 `torch.autograd.gradcheck`
   同样是「数值微分 vs 解析」；本件把同一族判据搬到 fp32 内核面。

**成本**：八件各取一个小形状（见 §4），逐件逐元素中心差分（每元素 2 次前向），加 2 个方向；
host 实测**整窗 < 1s**（见 §5.1 每件 wall），真机秒级。

**精度**：fp32 中心差分的相对误差下限约 `O(ε²·截断) 与 O(ufp/ε·舍入)` 的交点，实测最优
`rel ≈ 1e-7 … 2.7e-5`（§5.1），是阈值 `rtol=1e-4` 的 **4× 以上余量**；对被检的「×0.1% 级」偏差
（`rel≈1e-3`）判别力充足（§5.3）。故「够用」。

**归约面（沿 F2 纪律）**：解析侧的标量 `⟨out, dy⟩` 按**门面位宽 fp32** 累加（内核/闭式的归约面就是
fp32，F2 已把「归约不降位」钉死）；差分本身那次 `⟨out, dy⟩` 用 **fp64 累加只作量具**——它不参与
产品归约，只为把「两次前向的差」读到一位更准，避免 fp32 抵消噪声淹没梯度信号。

---

## 2. ε 选取表（逐件敏感性）

ε 是「截断误差（随 ε 增大而增大）↔ 舍入/抵消噪声（随 ε 减小而增大）」的折中。下表是
`reconcile/B3_probe_frob.py` 在 host(cpu) 上逐件扫 ε 得到的 **Frobenius 相对误差**（越小越好），
星号为本件采用值：

| op | ε=0.05 | ε=0.02 | ε=0.01 | ε=5e-3 | ε=2e-3 | ε=1e-3 | **采用 ε** | 依据 |
|---|---|---|---|---|---|---|---|---|
| linear | **1.93e-6\*** | 4.08e-6 | 9.13e-6 | 1.93e-5 | 4.73e-5 | 9.31e-5 | **0.05** | 线性项，曲率小，取大 ε 抗舍入 |
| add_ln | 3.07e-5 | **7.35e-6\*** | 1.43e-5 | 2.78e-5 | 6.46e-5 | 1.41e-4 | **0.02** | LN 的 `rsqrt` 曲率大，ε 过大截断升 |
| rope | **6.77e-7\*** | 1.61e-6 | 1.93e-6 | 9.09e-6 | 3.10e-5 | 3.60e-5 | **0.05** | 正交旋转，几乎无曲率，大 ε 最优 |
| attn_sw | 9.78e-6 | **8.51e-6\*** | 1.80e-5 | 3.54e-5 | 8.38e-5 | 1.82e-4 | **0.02** | softmax 指数项曲率中等 |
| gdn_delta | 1.09e-4 | 1.79e-5 | **1.18e-5\*** | 2.34e-5 | 5.50e-5 | 1.04e-4 | **0.01** | 递推连乘放大截断，需小 ε |
| gdn_conv | 3.11e-4 | 5.01e-5 | **1.59e-5\*** | 2.31e-5 | 3.95e-5 | 9.62e-5 | **0.01** | SiLU 非线性 + 4 抽头，曲率大 |
| readout | **5.39e-7\*** | 8.33e-7 | 8.98e-7 | 4.84e-6 | 1.41e-5 | 2.88e-5 | **0.05** | 纯搬运，精确，大 ε 最优 |
| lora_apply | **1.16e-6\*** | 2.73e-6 | 5.64e-6 | 1.11e-5 | 2.72e-5 | 5.49e-5 | **0.05** | 两次乘加，线性度高 |

要点（为什么每件不是同一个 ε）：`gdn_conv / gdn_delta / add_ln` 的**大 ε 侧先坏**（非线性曲率
使截断误差抬头），必须落到 `0.01–0.02`；`linear / rope / readout / lora_apply` 的**小 ε 侧先坏**
（取消噪声抬头），取 `0.05`。一张表把「逐件敏感性」钉成可复跑证据。

---

## 3. 为什么必须有 **INVALID** 语义（eager 伪装问题的机理）

### 3.1 机理
内核路径的入口是 `ascend_env.get_compiled(key, builder)`：命中缓存直返；后端不可用或 `builder`
抛异常时**返回 None**，入口层随即落 `_eager(...)`（`gemm_kernel.py:46-51` 一步典型）。这条回退
**不留编译键、常常也不记 blocker**（尤其 plan 拒形状那条）。

后果：`_eager` 与解析反传**数学同源、必然一致**，于是——
> 「内核回落 eager」会**伪装**成「内核 gradcheck 通过」。

这正是 M2 记录的「Python 层量不到内核内部」：只看梯度数值对不对，对「这一趟到底走没走内核」
**完全失明**。B_RESULT 已把这一点在 `train_step` 上举证（`KERNEL-USED-FAIL … why=blockers_present,
fallback_events`，`SYS1_KERNEL_USAGE_STRICT=1` 时退 5）。

### 3.2 三证绑定（本件实现口径）
每件对拍前先取「本趟增量」三证（**前向**口径，见下）：

```
① fwd_keys  = 前向这一步新编译出的 `<target>|` 键  ≥ 1     # 只有真编成才留痕，回退不留痕
② new_blockers 中不存在「未申报」项                       # 任何编译失败都必须在这本账上留字
③ fallback 公告里不存在「未申报」的键                      # 回落公告的实际发生（count_fallback_events 同源）
任一不满足 ⇒ INVALID（不是 PASS，也不是 FAIL）
```

**为什么键必须取「前向」增量、不能只数「有没有新键」**（本件设计过程中修掉的一个真洞）：
`linear` 的前向若被 `plan` 拒形状而静默落 eager，**反向**仍会走 `gemm_bwd_dw_kernel` 编出
`gemm_dw[...]` 键——只看「有没有键」会把这条假绿放进来（负例 N-b 抓到的正是它）。所以判据
锚在**前向新编译的键**上：前向没吃到内核，反向键再多也不作数。

**唯一已申报边界**（`DECLARED_BLOCKERS`）：
`(gdn_delta, cpu) → {"gdn_backward[cpu]"}`。机理：`_GdnFn.backward` 在 **cpu** target 上按设计走
torch 尺子并登记 `gdn_backward[cpu]`（`gdn_asc.py:724`，被 `test_ascend_gradcheck.py:319` 钉成断言）；
**ascend** target 反向已内核化（`BWD_STATUS="kernelized"`），无此边界。即：这是**已登记、已测试**
的设计分界，不是「未申报回落」，故不使该件 INVALID。其余 7 件在任何 target 下都不允许出现任何新增
blocker/回落。

> 与 KERNEL-USED 三证的**关系**：`autograd_asc.kernel_usage_report`（B 代理交付，见 `B_RESULT.md`
> §2.3）是**同一条三证**的纯函数版（`compiled_keys` 前缀 ∧ `blockers()=={}` ∧ `fallback_events==0`，
> 判决行 `KERNEL-USED-PASS kernels=<n> blockers=0 fallbacks=0 target=<t>`）。本件与其**同源同数**，
> 差别只有两处、且都是为 gradcheck 场景收窄：**(a) 取「前向增量键」而非全局键**（§3.2，反向后编键
> 不能顶前向的证）；**(b) 允许一份显式 `DECLARED_BLOCKERS` 白名单**（把 `gdn_backward[cpu]` 这条
> 已测试分界与「未申报回落」区分开）。B_RESULT 的 M1–M5 负例矩阵（拔键/伪造 blocker/无关 warning/
> 前缀错位）与本件负例 N-a/N-b 互补，共同把「假绿」这条路由堵死（R14）。

### 3.3 lora 的 `_eager_grad` 结构性边界（如实登记，不静默跳过）
`lora_asc._eager_grad`（禁改件）在 `Function.backward` 的 `no_grad` 语境里又调了一次
`torch.autograd.grad`，缺 `torch.enable_grad()`——于是**当 lora 门面落到 eager 反退路时**会抛
`RuntimeError: element 0 of tensors does not require grad`（B_RESULT §… 已把同一现象记在容器侧）。
本件处置：`run_one` 把「门面**未用内核**却抛」判 **INVALID**（这本就没量到内核，`reason=no-kernel`），
把「门面**已用内核**却抛」判 **FAIL**（红）。因此负例 N-a（桩 `backend_available=False`）里 lora
如实落在 INVALID，而 host 正跑（内核可用）时 lora 正常 PASS（§5.1）。这是**如实登记**，不是静默
跳过：机理、触发条件、处置都在此写明。

### 3.4 为什么 INVALID 不能折成 FAIL
INVALID = 「这本就没量到内核」——它既不是「数学错」（FAIL），也不是「数学对」（PASS）。把 INVALID
折成 FAIL 会让「环境/后端不在场」冒充「内核有缺陷」；折成 PASS 正是要堵的假绿。故独立成档，退出码
单列 `2`，且 `INVALID` 不计入 `pass`、不冒充 `fail`（SUMMARY 三个计数互斥）。

---

## 4. 形状选择（trap ②：先枚举 `plan()` 接受域，只测真可编形状）

每件一个小形状，**同时**满足「真机秒级」与「两 target 的 `plan()` 接受域」（否则测到的是
「plan 拒形状 → 静默落 eager」那条假路）：

| op | 形状 | 关键约束 | ascend 编译键（容器实证，§5.4） |
|---|---|---|---|
| linear | A(6,16) W(16,16) b(16) | n,k 落在块阶 16/32/64 | `gemm[ascend]\|n16k16b16x16x16anonedfloat32` |
| add_ln | x,r(6,16) g,b(16) | 二维等形、fp32 连续 | `add_ln[ascend]\|d16e1e-05b16` |
| rope | qkv(4,3,2,8) cos,sin(4,4) | packed、dim 偶、fp32 | `rope[ascend]\|h2d8s2g1qfloat32afloat32` |
| attn_sw | q,k,v(1,5,16) w=4 | dim%16==0、window≤seq | `attn_sw[ascend]\|h1d16w4b16x16` |
| gdn_delta | q,k,v(2,4,8) g(2,4,8) β(2,4) | 四路同头同长、dk/dv 常量 | `gdn[ascend]\|h2k8v8\|c8ubh` |
| gdn_conv | x(2,8,6) w(8,4) b(8) | 通道优先、窗宽 4 | `gdnconv[ascend]\|c8k4b2\|c8scalare` |
| readout | rows(8,16) ids=[3,0,3,7,5] | ids int、允许重复行号 | `gather[ascend]\|d16` |
| lora_apply | x(6,16) a(16,16) b(16,16) | r,n,k 均块宽整数倍 | `gemm[ascend]\|n16k16b16x16x16anonedfloat32` |

**rope 只把 qkv 当可微输入**（cos/sin 是常量）：`_RopeFn.forward` 经不透明内核使用 cos/sin，autograd
看不到它们被用（会报 `differentiated Tensor … not used in the graph`）；而 cos/sin 在训练里本就不
是待训参数，其梯度不在判决面。这是设计边界，随手记于此。

---

## 5. 自证留档（host + 容器）

### 5.1 host `--target cpu` 8/8 PASS（`dirs=2`）
```
$ .venv/bin/python ascend/gradcheck_card.py --target cpu
GRADCHECK-linear     PASS max_rel=2.004e-06 eps=0.05
GRADCHECK-add_ln     PASS max_rel=1.061e-05 eps=0.02
GRADCHECK-rope       PASS max_rel=6.865e-07 eps=0.05
GRADCHECK-attn_sw    PASS max_rel=9.131e-06 eps=0.02
GRADCHECK-gdn_delta  PASS max_rel=8.569e-06 eps=0.01   (fwd=gdn[cpu]|h2k8v8, 已申报回落 gdn_backward[cpu])
GRADCHECK-gdn_conv   PASS max_rel=2.653e-05 eps=0.01
GRADCHECK-readout    PASS max_rel=5.688e-07 eps=0.05
GRADCHECK-lora_apply PASS max_rel=1.133e-06 eps=0.05
GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8      # exit 0
```

### 5.2 三个负例（`reconcile/B3_probe_negatives.py`，`B3-NEG-DONE pass=3 fail=0`）
```
N-a 桩 ascend_env.backend_available=False ⇒ 八件全 INVALID reason=no-kernel（keys=0）   ✔
N-b 桩 gemm_asc.plan=None（plan 拒形状） ⇒ linear INVALID reason=no-kernel（非 FAIL）   ✔
N-c 解析参考 ×1.001             ⇒ linear FAIL max_rel=9.993e-04                          ✔
```
N-a 证明「eager 伪装」被三证拦住（数学是对的，但因没走内核判 INVALID，**不冒充 PASS**）；
N-b 证明「前向静默落 eager」被前向键判据拦住；N-c 证明对拍**真在量数**（咬住 ×0.1%）。

### 5.3 阈值 × 负例的分离度
正例最坏 `rel=2.65e-5`（gdn_conv）；负例 ×1.001 的 `rel≈1.0e-3`（逐件一致）。二者相隔约 **38×**，
阈值 `rtol=1e-4` 落在中间，判别干净。

### 5.4 容器 `cann910b-h` `--target ascend --no-run` 8 件编译键齐
```
$ python3 ascend/gradcheck_card.py --target ascend --no-run      # 容器，source CANN set_env + ASCEND_NPU_ARCH=dav-2201
GRADCHECK-linear  PASS … fwd_keys=[gemm[ascend]|n16k16b16x16x16anonedfloat32]
GRADCHECK-add_ln  PASS … fwd_keys=[add_ln[ascend]|d16e1e-05b16]
GRADCHECK-rope    PASS … fwd_keys=[rope[ascend]|h2d8s2g1qfloat32afloat32]
GRADCHECK-attn_sw PASS … fwd_keys=[attn_sw[ascend]|h1d16w4b16x16]
GRADCHECK-gdn_delta PASS … fwd_keys=[gdn[ascend]|h2k8v8|c8ubh]
GRADCHECK-gdn_conv  PASS … fwd_keys=[gdnconv[ascend]|c8k4b2|c8scalare]
GRADCHECK-readout   PASS … fwd_keys=[gather[ascend]|d16]
GRADCHECK-lora_apply PASS … fwd_keys=[gemm[ascend]|n16k16b16x16x16anonedfloat32]
GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8     # exit 0
```
即：八门面在昇腾后端**都能真编出前向内核**（无 plan 拒形状、无未申报回落）。ascend 轨的真数值
gradcheck 需真设备（本窗容器无卡，故 `--no-run` 只判编译面）。

---

## 6. 边界与未做项

- **bf16 不在范围**：C2 已证后端拒 bf16（本题沿用其结论）。
- **ascend 真数值 gradcheck**：需真设备；本件已备好 `--target ascend`（无 `--no-run`）轨，卡窗可直接跑
  （同一判据族）。本窗只验到 `--no-run` 编译面。
- **rope 的 cos/sin 不是可微输入**（§4），其梯度不在判决面。
- **attn_sw / gdn_conv 的反向是 fp32 闭式**（入口件无反向）：本件验的是闭式梯度正确性；其前向内核
  由三证锚定（`attn_sw[...]` / `gdnconv[...]` 键）。
- **未改任何产品件 / 判据**：本件只新建 `gradcheck_card.py` 与 `B3_probe_*.py`（+ 本文档）。

---

## 7. 复跑命令

```bash
# host：八件数值 gradcheck 判决（应 exit 0，SUMMARY pass=8）
cd release && .venv/bin/python ascend/gradcheck_card.py --target cpu

# host：三个负例自证（应 B3-NEG-DONE pass=3 fail=0）
.venv/bin/python ascend/kernels/reconcile/B3_probe_negatives.py

# host：ε 选取证据表（可选）
.venv/bin/python ascend/kernels/reconcile/B3_probe_frob.py

# 容器：ascend 编译键齐（应 exit 0，SUMMARY pass=8）
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && export ASCEND_NPU_ARCH=dav-2201 && \
  cd /work && python3 ascend/gradcheck_card.py --target ascend --no-run'
```
