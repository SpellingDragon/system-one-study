# F2_RESULT — p2-13 F2：产品位宽端到端（train_step `--dtype fp16`）

执行代理：F2（最后一处本地缺口）。日期：2026-10-11。
问题陈述出处：`reconcile/C2_dtype_matrix.md` §9 未闭环第 1 条——C2 已证 linear/dW/rope 三件
dtype 参数化且 fp32 9/9、fp16 9/9 **件级**闭环，但 `train_step.py` 全链 `out_dtype=torch.float32`、
参数默认 fp32 ⇒ **fp16 端到端未验**。本件补上：fp16 档接线、可判据化、本地四轨自证。

---

## 1. 实现（三件要事落了什么）

### 1.1 fp16 档端到端（`ascend/train_step.py`）
- CLI 新增 `--dtype {fp32,fp16}`（默认 fp32）；`DTYPE_BY_NAME` 不收 bf16（C2 §8 钉在 plan 期）。
- fp16 档：输入 `x0` 降 fp16（target 保持 fp32 考卷）；可训参数按**同 RNG 造 fp32 再统一降位**
  （`v.detach().to(dtype).requires_grad_(True)`，两档数字差只反映位宽不反映样本）；
  cos/sin 角度表**保持 fp32**＝C2 §7 生产混宽（键 `qfloat16afloat32`）。
- `_forward` 14 处门面调用统一改吃 `out_dtype` 参数（fp32 档传 `torch.float32`＝旧调用逐字）。
- loss 归约钉 fp32 面：`(logits.to(torch.float32) - target).pow(2).mean()`（fp32 档
  `.to(torch.float32)` 恒等返回 self，数值与改前逐字同）。
- **优化器位宽口径**（自行决策，申报见 §5-③）：fp16 档把**优化器状态与更新放 fp32 master
  副本**、step 后降回 fp16 leaves。一手依据：fp16 leaf 直喂 `torch.optim.Adam`，`eps=1e-8`
  舍入成 0、`v_hat` 在 |grad|≈7e-6 一档下溢 ⇒ step 出 ±inf/nan（首跑 `loss1=nan` 的根源，
  最小复现见 §3.2）。fp32 档分支逐字维持 `torch.optim.Adam(list(p.values()), lr=lr)`。

### 1.2 可判据化
- 新增判决行（**不动** NPU-REG / P2-STEP / KERNEL-USED 三行的格式与顺序；落 `main()`，
  `run()` 末行仍是 `P2-STEP-…`，tests 断言不受影响）：
  `DTYPE-TRACE <dt> params=<n> fp16=<n> fp32_accum=<true|false> fallback=<n>`
  （纯函数 `dtype_trace_line`，可喂假账自证）。
- 位宽纪律钉：`install/uninstall_reduction_guard`（run() fp16 窗口内罩 try/finally，对称计数）
  盯**Python 组装面**——fp16 张量不带 `dtype=fp32/fp64` 直接 `sum/mean/var/std/prod/amax/amin`、
  或任一操作数 fp16 的 `matmul/mm/bmm/__matmul__/__rmatmul__` ⇒ 记违规。违规>0 ⇒
  `DTYPE-TRACE-FAIL`（stderr）+ 退码 `EXIT_DTYPE_DISCIPLINE=6`（优先级 2>3>6>5，
  接线/下降判据保持在先，内核门禁排最后：归约落在粗尺子时"走没走内核"已无从谈起）。
  内核内部归约面不经 Python，由 C2 §2/§5 件级证据钉死——两段拼起来覆盖全风险面。
- KERNEL-USED 三证在 fp16 档同账本同判据同退码（host fp16 跑照样出 `KERNEL-USED-FAIL
  kernels=13 … fallbacks=1`，与 fp32 档同形；B 语义零改动，`kernel_usage_report` 等三符号未碰）。

### 1.3 位宽传导（`ascend/autograd_asc.py`，3 处最小改）
- `_LinearFn`：`ctx.bias_dtype` 记录 bias 原件位宽，backward 的 `db` 归约在 fp32 面做
  （既有 `.float().sum()`）、出口 `.to(ctx.bias_dtype)` 按输入位宽交货（fp32 恒等）。
- `_GdnFn.backward`：五梯（内核出口恒 fp32）交回 autograd 前按各自输入原件 `.to(dtype)`
  显式交货，不依赖引擎静默折算（实测：fp16 leaf 收 fp32 梯度会被引擎 cast、不报错——
  件上钉死比靠引擎诚实）。
- 其余 Function 不动：入口件契约本已齐（`add_ln/gdn/attn_sw/readout/lora` 的 `.float()`
  物化 + `.to(out_dtype)` 交货，C2 §5）。

## 2. 回归：fp32 档判决行逐字对照（before/after）

命令：`.venv/bin/python ascend/train_step.py --target cpu --device cpu`（stdout/stderr 分流捕获）

```
before (/tmp/f2_fp32_before.out)                after (/tmp/f2_fp32_after4.out)
NPU-REG skip not-imported target=cpu …         （逐字同）
P2-STEP-cpu loss0=0.910992 loss1=0.823897 \    （逐字同，数字含 6 位小数一致）
  reduced=True params_grad_ok=True
KERNEL-USED-FAIL kernels=13 blockers=1 \       （逐字同）
  fallbacks=1 target=cpu …['gdn_backward[cpu]']
+DTYPE-TRACE fp32 params=16 fp16=0 \           （唯一新增行，任务书允许的"新增判决"）
  fp32_accum=true fallback=1
```
`diff before after` ⇒ stdout 仅 `3a4` 一处（新增行）；**stderr 零差异**（RuntimeWarning 与
KERNEL-USED-BLOCK 行逐字同）；`EXIT=0` 两态一致。

## 3. 本地闭环自证（四轨，无卡）

### 3.1 轨 1：host cpu 直跑（数值与判据轨）
| 档 | 命令 | loss0 | loss1 | reduced | grad | DTYPE-TRACE | EXIT |
|---|---|---|---|---|---|---|---|
| fp32 | `…train_step.py --target cpu --device cpu` | 0.910992 | 0.823897 | True | 16/16 | fp16=0 fp32_accum=true fb=1 | 0 |
| fp16 | `… --dtype fp16` | 0.910987 | 0.823871 | True | 16/16 | fp16=16 fp32_accum=true fb=1 | 0 |

两档差：`|Δloss0|=5.0e-6`、`|Δloss1|=2.6e-5`（O(1) 量的相对差 ~3e-5，fp16 几个 ulp 量级，
合理；参数系同 RNG 降位构造，差只反映位宽）。**此为 host 数值，不作任何卡上结论**。

### 3.2 错误与修复记录（如实）
- 首跑 fp16 `loss1=nan`（EXIT=3）。定位：埋点复现显示 fwd#0 全链无 NaN/Inf、16 参 grad 齐且
  有限 ⇒ 源在 `opt.step()`。最小复现（非探针，验证性脚本）：
  `t=fp16 leaf, grad=6.8e-6 ⇒ torch.optim.Adam step 后参数 = [-inf,-inf,-inf,-inf]`；
  同梯度 fp32 面 step 正常（-0.0099853）。`torch.tensor(1e-8, dtype=fp16)=0.0`。
  → 处置：fp16 档 Adam 走 fp32 master（§1.1），fp16 轨转为全绿。
- guard 覆盖面首版不足：实测 `a.matmul(b)`（方法面）能抓、`a @ b`（运算符面，走 `__matmul__`
  类型槽）与 `torch.mm(...)`（C builtin 模块面）抓不到 → 补罩 `__matmul__/__rmatmul__`
  （F2_apply3，容器无关）；`torch.mm` 模块面**不补**（替换模块属性管不住 `t.mm()` 方法面，
  二者互斥复杂化收益低——组装面内积惯用写法是 `@`，已覆盖；如实申报 §5-③）。

### 3.3 轨 2：位宽纪律钉负例自证（`ascend/kernels/reconcile/F2_probe_dtype_guard.py`）
**8/8 PASS**（终态复跑）。要点：
- N1 负例：装 guard 伪造 `a.sum()`（fp16 直接归约）与 `a @ b`（fp16 内积）⇒ 违规 2/2 记账、
  `fp32_accum=false`；经 main 真链路（注入违规账）⇒ **exit=6** 且 `DTYPE-TRACE-FAIL` 落 stderr。
- N2 正例不误伤：`.float()` 升宽三连后违规增量 0；CLI 原链（fp32 档）exit=0 无扰。
- N3 显式 `sum(dtype=fp32)` 记合规计数 +1；N4 纯函数行格式逐字核；N5 卸载后不再记账（对称）。

### 3.4 轨 3：cpu-alias 全图 fp16 真编真跑（`F2_probe_fp16_host.py`，配方同 C2 alias 工装）
`TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/F2_probe_fp16_host.py`
⇒ **5/5 PASS**、exit 0：
- STEP：`loss0=0.910987 loss1=0.823871 reduced=True grad_missing=无`（target=ascend 路由）。
- AUDIT：`fp16=16 fp32_accum=true`（首轮此处 FAIL——抓到真缺陷，见 §3.5）。
- KEYS-ROUTE：10/10 件出键（桶名按 `compiled_keys` 真面：readout=gather/scatter_add、
  add_ln 反向=ln_bwd）；NO-BLOCKER：无回落。
- KEYS-DTYPE：`gemm…dfloat16`×3、`gemm_dw…dfloat16`×2、`rope…qfloat16afloat32`。
- 键分型：物化件键无位宽段（`add_ln|d32e1e-05b16` 等）＝C2 §5 既定形态，如实打印不判 FAIL。

### 3.5 组合暴露的既有缺陷（白名单单列，§4 偏离②）
alias 首轮 `fp32_accum=false`，violation 原文：
`['__matmul__(fp16 内积面: (16, 32) × (32, 2))', '__matmul__(fp16 内积面: (16, 2) × (2, 32))', …]`
源＝`gemm_kernel._eager` 回落线（beta 的 linear n=heads=2 小形状被 gemm 模具拒绝 → eager），
原式 `acc = (A @ W.T).float() + bias.float()` 的**乘加在 A 的位宽面做、算完才升宽**，与本函数
doc"先转 fp32 乘加"不符；fp32 轨因入口 `to32` 升宽从未暴露，**train_step fp16 组合才现形**
——正是任务书预告的"某件 fp16 在训练步组合下才暴露缺陷"。修复（单行）：
`acc = A.float() @ W.T.float() + bias.float()`（fp32 时 `.float()` 恒等 ⇒ 既有行为逐字不变）。
全库 grep 同类模式仅此一处（`grep -rn "A @ W|@ W.T" ascend/kernels/*.py`）。
修复后 alias 轨/容器轨 guard 均 `fp32_accum=true`，C2 矩阵 fp32 9/9 + fp16 9/9 复跑不破。

### 3.6 轨 4：容器 `cann910b-h` 编译腿（`F2_probe_card_fp16_step.py`，C2 card 桩配方）
前置：compat 同步——容器 `dist-packages/…/port910b_compat.h` 为旧版 `d0547c0b`，按 C2 §7/S 配方
备份至容器 `/tmp/port910b_compat.h.f2bak` 后覆盖 trunk `/tilelang/src/…`（md5
`ca023c2e01746ca52c87f36bcea93ee9`，两侧复核一致）。缓存用 `TILELANG_CACHE_DIR=$(mktemp -d)`
（跑后容器盘 `df /` 仍 5.1G avail，零增长压力）。命令：
```
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/F2_probe_card_fp16_step.py'
```
⇒ **4/4 PASS**、exit 0（判编译与键账；**桩轨不判数值**，口径同 C2 card 腿——no-op 发射下
空表不当证据）：
- COMPILE：全图 `compiled=13 src=40035B 编译失败=无`。
- KEYS-ROUTE：10/10 件出键、blockers 无；KEYS-DTYPE：`dfloat16` 三处齐 + `qfloat16afloat32`。
- 生成码位宽旁证（`__global__` 签名原文）：
  `extern "C" __global__ __vector__ void gemm_impl_kernel(__gm__ half* A, __gm__ half* C, __gm__ half* W, __gm__ float* bias, int32_t m)`
  ——A/C/W 落 half 面、bias 恒 float 面，"主干 fp16、累加/偏置 fp32"在真 bisheng 产物上的投影。
- AUDIT：`fp16=16 fp32_accum=true`，step_err=无（fp16 档训练步在真方言路由下完整走完）。

## 4. 回归总表（改后全部复跑）

| 项 | 命令 | 结果 | 基线 |
|---|---|---|---|
| host 单测 | `pytest tests -q -m "not integration"` | **340 passed 0 failed**（3 skipped 既有：NPU 需求×1、MPS 禁用×2） | 一致 |
| C2 件级矩阵 fp32 | `C2_probe_dtype_matrix.py --dtypes fp32` | 9/9 | 一致 |
| C2 件级矩阵 fp16 | `… --dtypes fp16` | 9/9 | 一致 |
| B·H1 设备门 | `B_probe_h1_npu_reg.py` | B-H1-DONE PASS（退码 4 语义在位） | 一致 |
| B·内核门禁负例 | `B_probe_kernel_usage_negative.py` | B-KU-DONE PASS（退码 5 语义在位） | 一致 |
| B·cardrun 判决矩阵 | `B_probe_cardrun_verdict_matrix.py`（**不带** CARDRUN_SRC） | **33 ok / 0 MISS / exit 0** | 一致¹ |
| train_step cpu 两档 | §3.1 | fp32 逐字回归 + fp16 全绿 | 本节新增 |
| card_run.sh 未碰 | `bash -n` OK；md5 `cf0c2bdf…` | B 写面零改动（禁改自证） | — |

¹ 踩坑记录：首跑误用 `/tmp/b_m1.sh`（B 会话遗留的 **sabotage 变异体**，`CARDRUN_SRC` 只为负例
自证留口）⇒ S2 两例 MISS 是变异体被抓住的表现，非 F2 破坏；用正身重跑即 33 ok。

## 5. 自行决策与偏离申报

**③ 疑惑点与自行决策**
1. **"梯度累加 fp32"落点**：按权威口径拆两半——反向面归约/内积恒 fp32（guard 盯组装面 + C2
   件级证据盯内核面 + 入口件显式 `.to()` 交货钉位宽）；**优化器面走 fp32 master weights**
   （§3.2 一手证据：fp16 直喂 Adam 必炸）。`.grad` 叶子存储仍是 fp16（autograd 累加面在本件
   每步清零、无跨步 fp16 累加发生）；正式训练如需 fp32 梯度累加器属上层设计，不在本件射程。
2. **dtype 判法**（任务书"判法自选但要可解释"）：Python 层方法/运算符面包装 + 内核面引 C2 件级
   证据的两段拼合；口径与注释写死在 guard 段头。
3. **`torch.mm` 模块面不罩**：理由见 §3.2；`@` 面与 `.matmul()` 面已覆盖组装面惯用写法。
4. **bf16 不放开**：入口与 CLI 同守 C2 §8（plan 期显式拒），无自行放宽。
5. **readout 桶名**：探针按 `compiled_keys()` 真名（gather/scatter_add）断言，非臆造。

**④ 偏离记录**
1. **白名单字面外的一处必要改动**：`ascend/kernels/gemm_kernel.py` 单行（不在
   `*_asc.py` 模式内）。任务书条件"某件 fp16 在训练步组合下才暴露缺陷时才动，并在战报单列"
   实质成立（§3.5 全过程+复跑证据），按"意图优先"执行；fp32 行为恒等由 C2 fp32 9/9 +
   pytest 340 + fp32 判决行逐字同三重回归钉住。**如编排者判定应回滚，改回原式即可**
   （原式已注释留存），但回滚后 fp16 档纪律钉必红（exit 6），fp16 端到端即不可宣称。
2. **新建文件名**：`F2_apply.py`/`F2_apply2.py`/`F2_apply3.py`（一次性锚点补丁脚本）不完全匹配
   `F2_probe_*.py` 模式。保留理由：内含 before/after 锚点原文，是"fp32 档逐字不变"承诺的
   可审计凭据；未改名以免混淆 B/C2 探针命名系。
3. **容器 compat 覆盖**：`dist-packages` 为容器环境（不在仓库禁改名单），按任务书既有配方执行，
   旧件备份 `/tmp/port910b_compat.h.f2bak` 可回退。

## 6. 产物清单

| 文件 | 性质 |
|---|---|
| `ascend/train_step.py` | 改（382→558 行：--dtype、guard 段、DTYPE-TRACE、master-weights、退码 6） |
| `ascend/autograd_asc.py` | 改（587→597 行，3 处位宽传导；B 三报告符号未碰） |
| `ascend/kernels/gemm_kernel.py` | 改（1 行缺陷修复 + 注释，见 §5-④-1） |
| `ascend/kernels/reconcile/F2_probe_dtype_guard.py` | 新建（负例自证 8/8） |
| `ascend/kernels/reconcile/F2_probe_fp16_host.py` | 新建（cpu-alias 全图 5/5） |
| `ascend/kernels/reconcile/F2_probe_card_fp16_step.py` | 新建（容器编译腿 4/4） |
| `ascend/kernels/reconcile/F2_apply{,2,3}.py` | 新建（锚点补丁脚本，审计凭据） |
| `reconcile/F2_RESULT.md` | 本文 |

改后 md5：train_step `94cd8e2d…`、autograd_asc `26d9fd1a…`、gemm_kernel `d06e5cb8…`
（改前备份：`/tmp/f2_before_{train_step,autograd_asc}.py`，md5 `9be24b43…`/`9aa2a7ea…`）。

## 7. 留给卡窗的（本件不能替卡说话的部分）

- fp16 档**卡上数值**（真设备 `--target ascend --dtype fp16`：loss 轨迹、`half` 面数值与
  host 差异）——本地只证了"真方言编得出、键含位宽段、组装面纪律干净"。
- fp16 档 `KERNEL-USED` 在 strict 门禁（`SYS1_KERNEL_USAGE_STRICT=1`）下的卡窗判决——
  cpu 轨的 `gdn_backward[cpu]` 回落与位宽档无关，卡窗应归零。
- 若卡窗要 fp16 档一跑到底：`card_run.sh` 现调 train_step 不带 `--dtype`（默认 fp32 行为，
  B 写面 F2 未动）；卡窗加档属编排者决定（一行参数即可，判决行格式已兼容）。
