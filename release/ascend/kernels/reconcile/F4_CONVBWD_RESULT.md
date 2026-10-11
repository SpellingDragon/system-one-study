# F4 · GDN 短卷积反向去 `F.conv1d` 依赖（P1-1zb）结果

> 执行代理 F4｜波次 p2-13｜**不占卡**｜本窗只做 host 数值对拍 + 容器（cann910b-h）编译判决；
> **卡上真机判决本窗不宣称**（`GRADCHECK-gdn_conv` 卡上 PASS 需下窗一枪）。

## 0. 结论

`gdn_conv` 反向**已不再调 `torch.nn.functional.conv1d`**：昇腾 target 走新的 910B 标量面自研件
`conv_bwd_asc_impl`（翻转核加权 gather + 抽头/偏置归约），其余 target 落同口径纯 torch 尺子
`_bwd_eager`（同样零 conv1d）。接口签名与张量 I/O 不变；`target=cpu` 语义路不动；`kernel_usage_report()`
三证语义未变。host 判据件 8/8、全量回归 340 passed、容器编译腿 8/8 + 反向件真编 PASS。

## 1. 改动清单（白名单内）

| 文件 | 前 md5 | 后 md5 | 改动 |
|---|---|---|---|
| `ascend/kernels/gdn_conv_asc.py` | `329779794394ebbda3c1ea3be4ab750e` | `28cc045ad9064a59d093ca73714303f9` | 新增 `conv_bwd_asc_impl` / `_bwd_eager` / `plan_backward` / `run_backward` / `conv_backward`；`_eager` 去 conv1d；反向段 docstring |
| `ascend/autograd_asc.py` | `26d9fd1a8514b84335a513c1264080b1` | `92e2fd0e6f7c251bf23b81b4afee64f7` | `import gdn_conv_asc`；`_GdnConvFn.backward` 改调 `conv_backward`；`_conv_silu_torch` 去 conv1d + 前导维无关；docstring |

新建（探针/结果）：`ascend/kernels/reconcile/F4_probe_convbwd.py`、`F4_probe_convbwd_compile.py`、本文件。
**未改**：`run_numerics.py` / `compile_all_asc.py` / `gradcheck_card.py` / `run_delta_card.py` /
`train_step.py` / `card_run.sh` / `window3_card.sh` / trunk `port910b_compat.h` / `patches/*` /
`bench_card.py` / `reconcile/{D3,E3,S3,U2,V2,C2,C4,B3}_*` / `runs/**` / 任何 `tasks.md`。**未用 git**。

## 2. 改法（实现选择）

- **昇腾反向件** `conv_bwd_asc_impl`：与前向 `conv_asc_impl` 的 `scalar` 档**同族**（`T.Kernel(cores)` +
  `T.Vector()` + 纯 `T.serial` + `T.alloc_var`，零 `T.copy`/SimtVF/Parallel，避开该代际无 SIMT 模型与
  MTE2/MTE3 未证真面）。两趟：趟 1 逐 (b,t) 算 `dz=dy·silu'(z)`（`silu'(z)=sigmoid(z)·(1+z·(1−sigmoid(z)))`），
  顺手把 `DB[c]=Σ dz`、`DW[c,j]=Σ dz·x_{t−pad+j}` 归约；趟 2 `DX[b,c,s]=Σ_j W[c,j]·dz_{s+pad−j}`。
- **不落 GM 暂存**（不赌 G-F3 跨条目读写可见性）：趟 2 的 `dz` 现算现用；抽头账只在核内 UB 累加器
  `dw_ub[k]`（被读也被写，非 G-C8 死缓冲）里累，出口一次性写回 `DW`。
- **边界**：左端 `idx>=0` 补零（地址先夹到 0，不出负下标）；趟 2 读"未来格"先 `tcl=T.min(t,length-1)`
  夹下标再用 `T.if_then_else(t<length, …)` 掩掉，绝不出上界。
- **位宽纪律（F2 口径）**：内核全程 fp32；入口 `conv_backward` 在主干 fp16 时**显式** `.float()` 升宽进
  内核、出口按输入原件位宽 `.to()` 交货，**不靠引擎静默折算**；fp64 不进内核（只在纯 torch 尺子上，供 gradcheck）。
- **键**：`ascend|gdnconvb[ascend]|c{C}k4b{B}|c{cores}scalar`（不含动态长度 L，与前向同口径）。
- **blocker 策略**：`gdn_conv_backward[ascend]` 仅在昇腾内核未编成时登记（R14 防假绿）；cpu 不登记
  （保持原静默，不破坏既有 `KERNEL-USED` 行）。

### 实现选择：两条备选方案与未采用理由

- **备选一（未采用）：反向里手工重算整条前向再借 torch 自动微分** —— 这正是旧版 `F.conv1d` 的写法思路
  （前向 `pad→conv1d→silu` + autograd 回传），NPU 上不可执行（`SetPrecisionMode…500001`，E3 判真缺陷）；
  且它把数值正确性寄托在被判有缺陷的那条 op 上，等于原地打转。
- **备选二（未采用）：把反向也塞进前向模具** —— 前向出口是 `y=SiLU(z)` 不是 `z`，SiLU 链式需要的
  pre-activation（`z`）没留痕；要么多存一份整表（GM 暂存 → 又撞 G-F3），要么在反向里重算整条前向
  （= 备选一）。二者都劣于"独立反向件 + UB 累加器"。

## 3. host 双证（必须真跑）

### 3.1 判据件（`gradcheck_card.py --target cpu`）

```
GRADCHECK-gdn_conv PASS max_rel=2.653e-05 eps=0.01
    · fwd_keys=[gdnconv[cpu]|c8k4b2] wall=0.04s
GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8
```

- gdn_conv 新 `max_rel` = **2.653e-05**，与旧值 **逐字相同**（旧 `2.653e-05`）⇒ 新反向与旧闭式数值等价，
  未变差，**未调阈值**。
- 8/8 PASS，`kernel_used=8/8`（三证语义未破）。

### 3.2 host 独立对拍（`F4_probe_convbwd.py`，15/15）

```
T1 产品反向(cpu) vs 独立参考（定义式 conv1d+自动微分）— fp32
  dx rel=2.801e-08  dw rel=8.954e-08  db rel=1.087e-07   (全 < 1e-5; dtype 全 fp32)
T2 产品反向 fp16 档：出口按输入位宽交货（不静默升宽）
  dtype 全 fp16；dx rel=4.922e-04  dw rel=4.124e-04  db rel=4.348e-04  (全 < 2e-3)
T3 昇腾内核算法镜像 vs 产品 _bwd_eager（同一套账）
  dx rel=2.707e-08  dw rel=9.378e-08  db rel=1.180e-07   (全 < 1e-5)
T4 负例（见 §5）
F4-CONVBWD-DONE pass=15 fail=0
```

- T1 参考是**独立**的（探针内用 `F.conv1d` 定义式现场自动微分，非产品路径）；产品反向（cpu→`_bwd_eager`）
  与其 fp32 相对误差 1e-8 量级。
- T3 把 `conv_bwd_asc_impl` 的两趟用 torch 标量面**逐字复刻**（含"先夹后掩"），与产品 `_bwd_eager` 对拍
  1e-8 量级 ⇒ 证明"内核设计"与"已证尺子"是同一套账，方言翻译之外无额外逻辑。

## 4. 容器编译腿（cann910b-h；compat 安装版 md5=`ca023c2e`）

- 环境：`source set_env.sh` + `BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler` +
  `ASCEND_NPU_ARCH=dav-2201`；`TILELANG_CACHE_DIR=/dev/shm/f4*`；Python 3.10.12；`/work` = release 挂载。
- compat（install）`/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`
  md5 = **`ca023c2e01746ca52c87f36bcea93ee9`** ✓
- 容器内两产品件 md5 与 host 一致（`autograd_asc.py=92e2fd0e…`、`gdn_conv_asc.py=28cc045a…`）。

### 4.1 `gradcheck_card.py --target ascend --no-run`

```
GRADCHECK-gdn_conv PASS max_rel=- eps=0.01 keys=1 mode=no-run
    · fwd_keys=[gdnconv[ascend]|c8k4b2|c8scalare]
GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8
```

8/8 编译键齐。

### 4.2 单独真编译**新的反向件**（`F4_probe_convbwd_compile.py`）

```
F4-CARDBWD-fwd PASS keys=1 first=ascend|gdnconv[ascend]|c16k4b2|c8scalare .o~1864B blockers=0
  · artifact kernel.aibin 6400B  device_kernel.asc 1864B  executable.so 25192B
F4-CARDBWD-bwd PASS keys=1 first=ascend|gdnconvb[ascend]|c16k4b2|c8scalar .o~4898B blockers=0
  · artifact kernel.aibin 6416B  device_kernel.asc 4898B  executable.so 33392B
F4-CARDBWD-SUMMARY pass=2 fail=0
```

反向件 `keys=ascend|gdnconvb[ascend]|c16k4b2|c8scalar`、`src~=4898B`、设备件 `kernel.aibin=6416B`、
可执行 `executable.so=33392B`；blockers=0（未静默回退）。前向件同批对照 PASS。

> 注：`compile_all_asc.py` 行里的 `.o~<bytes>` 实为 `len(get_kernel_source())`（源长），本节另列真实
> 工件字节数，二者都在。`executable.so`/`kernel.aibin` 为容器缓存目录中该核新增产物的真实大小。

### 4.3 `compile_all_asc.py --target ascend --products`（产品腿 4/4）

```
ASC-PROD-linear_prod PASS Y-linear_prod-ASC-COMPILE-PASS keys=1 src~3004B track=card(no-run)
ASC-PROD-dw_prod     PASS Y-dw_prod-ASC-COMPILE-PASS     keys=1 src~2803B track=card(no-run)
ASC-PROD-readout_prod PASS Y-readout_prod-ASC-COMPILE-PASS keys=2 src~942B track=card(no-run)
ASC-PROD-addln_prod  PASS Y-addln_prod-ASC-COMPILE-PASS  keys=1 src~3652B track=card(no-run)
—— 产品腿 PASS=4 FAIL=0 | 对照腿 PASS=8 FAIL=0 SKIP=0 —— 门禁失败数=0
```

接口腿 9/9 PASS（含 `ASC-gdn_conv PASS .o~1864B keys=1 first=ascend|gdnconv[ascend]|c16k4b2|c8scalare`）。

## 5. 负例自证（防"改完看起来对"）

向新反向注入两处错，经**判据件** `gradcheck_card.run_one("gdn_conv","cpu")` 复跑（monkeypatch `_bwd_eager`）：

| 用例 | 判决行 | 结论 |
|---|---|---|
| before（正身） | `GRADCHECK-gdn_conv PASS max_rel=2.653e-05 eps=0.01` | 正身绿 |
| N1（dx 抽头**方向**写反，不翻转） | `GRADCHECK-gdn_conv FAIL max_rel=7.479e-01 eps=0.01` | 必红 ✓ |
| N2（**dw 漏掉最后一个抽头**） | `GRADCHECK-gdn_conv FAIL max_rel=4.663e-01 eps=0.01` | 必红 ✓ |
| after（复原） | `GRADCHECK-gdn_conv PASS max_rel=2.653e-05 eps=0.01` | 复原绿 ✓ |

⇒ 判据件对"方向错""漏项"两类错都能当场抓出（相对误差从 1e-5 跳到 ~1e-1 量级），
"改完看起来对"骗不过判据。

`F.conv1d` 依赖清除（前后对比）：

```
# BEFORE（旧）  F.conv1d( 调用点：
ascend/autograd_asc.py:455   F.conv1d(...)          # _conv_silu_torch 反向闭式
ascend/kernels/gdn_conv_asc.py:315  F.conv1d(...)   # _eager 回退尺子

# AFTER（新）  grep -n "F\.conv1d(\|conv1d(" ascend/autograd_asc.py ascend/kernels/gdn_conv_asc.py
（空：零调用点；仅剩解释"为何不调 conv1d"的 docstring 文本）
```

## 6. 回归

| 项 | 判决行 | 结果 |
|---|---|---|
| `pytest tests -q -m "not integration"` | `340 passed, 3 skipped, 3 deselected` | 340/340 ✓ |
| `compile_all_asc.py --target cpu` | `PASS=9 FAIL=0` 门禁失败数=0 | 9/9 ✓ |
| `compile_all_asc.py --target ascend --products`（容器） | 产品腿 PASS=4 FAIL=0 | 4/4 ✓ |
| `run_numerics.py --only linear_prod --track cpu-alias` | `Y-linear_prod-PROD-PASS rel=2.54e-07 dir_disc=1.37e+00 reuse=4/4 relu_neg_clipped=802 shapes=5 track=cpu-alias` | 与改前逐字相同 ✓ |

（回归期发现并修复一处自身回归：`_conv_silu_torch` 原先只吃 3D，`gradcheck` 传 2D `(C,L)` 时 `IndexError`；
已改为前导维无关（内部展平再收形），复跑 340 passed。）

## 7. 缺口与下窗待办

- **卡上真机判决本窗不宣称**：`GRADCHECK-gdn_conv` 的卡上 PASS（`--target ascend` 真设备数）需下窗一枪；
  本窗只交"容器真编 PASS + host 数值等价"。
- **fp16 归约面**：内核全程 fp32 归约（符合 F2 位宽纪律）；fp16 出口按输入位宽交货，T2 已证出口 dtype 为
  fp16 且相对误差 4~5e-04（< 2e-3）。
- 未用 GM 暂存（刻意规避 G-F3）；若日后要压 UB，可评估，但当前实现已满足"去 conv1d + 数值等价"目标。
