# Y_RESULT — P1-1i 判决件改接产品件（代理 Y）

- **变更**：`teacher-p2-production-full` / `p2-13-ascend-runtime`（嵌套子变更，openspec CLI 不索引，按调用方下达的任务文本直读）
- **孙任务**：**P1-1i**——消除元缺陷「真机判决件量的不是产品代码」
- **本窗性质**：**本地 + 容器（有 bisheng 无设备）**；**不含真机判决行**（下窗产生，见 §5）
- **改动件**：`ascend/port910b/run_numerics.py`、`ascend/compile_all_asc.py`、`ascend/port910b/card_run.sh`（白名单三件，未越界；产品件/入口件/`ascend_env.py`/`autograd_asc.py`/`train_step.py`/trunk/tests **一字未动**）
- **一句话结论**：三条产品件 case（`linear_prod`/`dw_prod`/`readout_prod`）已上轨并**带假绿守卫**；host cpu 方言别名轨真编真跑数值全绿（rel 最大 `2.54e-07`）；容器 ascend 轨三件 `Y-*-ASC-COMPILE-PASS` + 九件 `ASC-* PASS`（target=ascend，零 PRECOMPILE-ERR）；`card_run.sh` 的 overlay 落点按 `-I` 根修正并加存在性断言（负例实测 `exit 1`）。**下窗跑这份工装，量的就是产品代码。**

---

## 0. 结论摘要（三句话，均可复核）

1. **错位①（判决件不对齐）已闭合**：`run_numerics.py` 原有四条 attempts 内联 case 改名 `*_intrinsic`
   并标 `track=performance`（**正文与判据一字未动**，容器复跑 `NUM-gemm_l1_intrinsic-PASS`/
   `NUM-readout_intrinsic-PASS` 证明改名没改坏历史判决），新增三条**经对外入口件**
   （`gemm_kernel`/`gemm_bwd_dw_kernel`/`letter_readout_kernel`）+ 走 `ascend_env` 路由的产品件 case。
2. **假绿守卫是 load-bearing 且已双向证明**：正向——host 探针实测「只改 `sys.modules['tilelang.ascend.language']`
   不改父包属性」时产品件**静默落 torch eager 且 rel=0.0**；反向——`scratch/y_guard_proof.py` 故意还原
   `compile_kwargs` 后守卫如实抛 `Y-linear_prod-PROD-FAIL AssertionError: 假绿守卫…`。本窗它在容器里
   **真实拦住了一次**（缺 `BISHENG_HOME` ⇒ 落 eager ⇒ 守卫报 FAIL，而不是交出假绿数字）。
3. **错位②③已修**：`compile_all_asc.py` 九条腿一律只走入口件（`plan(..., n=, k=)` 那条第二套口径删除），
   `--target` 可切（缺省 cpu）；`card_run.sh` 桩头落点改为 `$PIP_SRC=$(tilelang)/src`（旧值多了一层
   `tl_templates`，这就是 wave2 六件 `'c_api/asc_simd.h' file not found` 的根因），并加起跑前存在性断言。

---

## ① 完成情况（逐条附可复跑凭据）

| 子项 | 状态 | 凭据（可复跑命令 → 真实输出） |
|---|---|---|
| 复核错位①（内联正文≠产品正文） | ✅ | 逐条比对 `num_readout` 内联的 `attempts/B/b_readout_910b.py::build()`（scores 投影，含点积）vs 产品 `letter_readout_asc.py`（纯 gather/scatter_add）；`num_gemm_l1` 内联 `alloc_l1/asc_copy_*` cube 路 vs `gemm_asc.gemm_scalar_body`（零 cube） |
| 复核错位②（`plan()` 签名不符） | ✅ | 已修前实测复现：`ASC-gemm PRECOMPILE-ERR TypeError: plan() got an unexpected keyword argument 'n'`（旧 host 跑，PASS=0 FAIL=7） |
| 复核错位③（overlay 落点错一层） | ✅ | 机制链：`patches/common.h:11` `#include "c_api/asc_simd.h"` ⇒ 按 `-I` 根 `tilelang/src` 解析；旧 `PIP_SRC=$(dirname "$TPL")`=`tilelang/src/tl_templates` ⇒ 找不到（`bundle_ondemand.sh:37` 的 `$(dirname "$TPL")/../$d` 是正确形态） |
| run_numerics：内联 case 改 performance-track | ✅ | `python3 run_numerics.py --only gemm_l1` → `[note] …弃用别名 ⇒ gemm_l1_intrinsic…`；容器 `--only gemm_l1_intrinsic --no-run` → `NUM-gemm_l1_intrinsic-PASS rel=n/a (compile-only) track=performance`（EXIT=0） |
| run_numerics：新增三条产品件 case + TARGETS/`--only` | ✅ | host：见 §2 三行 `Y-*-PROD-PASS`；`--only nope` → `NUM-nope-FAIL unknown target；可用：linear_prod,dw_prod,readout_prod,…`，`UNKNOWN-TARGET-EXIT=1` |
| readout 产品 case 用 U 判据口径 | ✅ | `rel<1e-5` + 逐位同（`bitexact=True`）+ 散取方向判别（>0.5）+ 累加/覆盖口径判别（>0.3）+ gather/scatter 双键守卫；容器 `keys=2` |
| compile_all_asc 修 gemm 腿 + 逐件按产品接口取判决 | ✅ | host：`ASC-gemm PASS .o~2088B keys=1 first=cpu|gemm[cpu]|n32k32b16x32x32anone`；`—— 汇总 target=cpu PASS=9 FAIL=0 ——` |
| card_run.sh overlay 路径 + 断言 + [3] 产品件优先 + [4] 保持 | ✅ | `bash -n` OK；正例 `OVERLAY-OK headers=5 tpl=6` exit 0；负例 `FATAL overlay-missing: …/src/c_api/asc_simd.h` **NEG-EXIT-CODE=1** |
| 容器产品件 ascend 编译判决 | ✅ | `Y-linear_prod-ASC-COMPILE-PASS keys=1 src~3004B` / `Y-dw_prod-… keys=1 src~2803B` / `Y-readout_prod-… keys=2 src~942B`；另九件 `--target ascend` `PASS=9 FAIL=0` |
| 真机判决行 | ⛔ 本窗不产 | 无卡（纪律：勿试）；本任务边界=「下窗量的是产品件」。见 §5 |

---

## ② 数值凭据（host cpu-alias 轨：生产 ascend 正文真编真跑）

```
cd release && .venv/bin/python ascend/port910b/run_numerics.py --only <case> --track cpu-alias
Y-linear_prod-PROD-PASS  rel=2.54e-07 dir_disc=1.37e+00 reuse=4/4 relu_neg_clipped=802 shapes=5 track=cpu-alias
Y-dw_prod-PROD-PASS      rel=2.21e-07 dir_disc=1.38e+00 reuse=4/4 shapes=4 track=cpu-alias
Y-readout_prod-PROD-PASS rel=0.00e+00 fwd=0.00e+00 bwd=0.00e+00 bitexact=True reuse=6/6 shapes=5 track=cpu-alias
```

阈值口径：`REL_TOL_LIN/DW=2e-5`（同 `S_cpu_oracle.py`）、`REL_TOL_READOUT=1e-5`（同 `U_oracle.py`）、
`WRONG_DIR_MIN=1e-2`、`DISC_MIN=0.5`、`DISC_MIN_SCATTER=0.3`（本波实测定标：覆盖式错参考在 dup-heavy
子例=0.46、odd-dim=0.39、dup-dominant≈0.8，而 0.5 那把尺来自"全表级"错参考，对只落重复行的错过紧 ⇒
单列下限并留 1.3x 余量，正参考恒 rel=0 故判别比 ≥1e4）。
`reuse=N/M` = `compile_count == distinct(形状键)` ⇒ 证明 m 真落在动态上界上（intrinsic case 是常量形状，永远量不到这条产品行为）。

## ③ 容器凭据（有 bisheng 无设备；§12 compat 与 trunk 已核对为同一份，`diff -q` = SAME）

```
产品件只判编译（`--track card --no-run`，冷缓存各 4–7s）：
  Y-linear_prod-ASC-COMPILE-PASS keys=1 src~3004B track=card(no-run) note=无设备⇒只判编译
  Y-dw_prod-ASC-COMPILE-PASS     keys=1 src~2803B …
  Y-readout_prod-ASC-COMPILE-PASS keys=2 src~942B …        # gather + scatter_add 两份正文都在
九件产品接口 ascend 编译：python3 ascend/compile_all_asc.py --target ascend
  ASC-gemm PASS ascend|gemm[ascend]|n32k32b16x32x32anone / ASC-gemm_bwd_dw PASS ascend|gemm_dw[ascend]|n32k32tc16b32x32
  ASC-add_ln / ASC-rope / ASC-attn_sw / ASC-gdn / ASC-gdn_conv / ASC-letter_readout / ASC-lora 全 PASS
  —— 汇总 target=ascend PASS=9 FAIL=0 ——        （旧 harness 的 gemm PRECOMPILE-ERR 已消失）
```

---

## 4. 三档 track 语义（同一套产品件 case，靠 `--track`/`--no-run` 分派）

| track | 触发 | 递进去的 target | 产出 | 说明 |
|---|---|---|---|---|
| `card` | 有 npu（下窗） | `"ascend"` | `Y-<case>-PROD-PASS`（真机数值判决） | 什么都不桩，真编真发射 |
| `card` | 无 npu（容器） | `"ascend"` | `Y-<case>-ASC-COMPILE-PASS` | cpu 张量 + 桩发射；**只判编译，不冒充数值判决** |
| `cpu-alias` | host 缺省（auto） | `"ascend"`（路由/键名全是产品件的） | `Y-<case>-PROD-PASS rel=` | `tilelang.ascend.language` 的 `sys.modules` + **父包属性两处一起**指向 cpu 方言 + `compile_kwargs` 翻成 c 后端三件套 ⇒ 无卡能拿到的最强证据 |

手法出处 `reconcile/U_oracle.py::run_prod`，**本波修正版**：`import a.b.c as T` 编译为 `getattr(parent,'c')`，
只改 `sys.modules` 不生效 ⇒ 少改一处就是「静默 eager + rel=0.0 假绿」。

---

## 5. 下窗该看哪些判决行（本任务的交付面）

`bash card_run.sh` 的 [3] 段现在**先跑产品件**：`CASE-PROD-linear_prod|dw_prod|readout_prod`
（grep `Y-<case>`），后跑对照轨 `CASE-PERF-*`（grep `NUM-<case>`）。
**只有 `Y-*-PROD-PASS rel=` 才是产品结论**；`NUM-*-PASS` 属 §12/intrinsic 性能对照，不可当正确性判决引用。
[2] 段若桩头没落对位置 ⇒ `FATAL overlay-missing: …` 直接 `exit 1`（不会再烧掉整窗）。
**旧名兼容（白名单外的两支脚本没断）**：`fixup_card.sh:8` / `bundle_ondemand.sh:47` 仍用
`gemm_l1 dw addln readout` 旧名 + `grep -E "NUM-$t"`。弃用别名把旧名转到 `*_intrinsic`，判决行
`NUM-gemm_l1_intrinsic-…` 仍以 `NUM-gemm_l1` 开头 ⇒ grep 形态照样捞到（host 实证四条：
`NUM-gemm_l1_intrinsic-FAIL` / `NUM-dw_intrinsic-FAIL` / `NUM-addln_intrinsic-FAIL` /
`NUM-readout_intrinsic-FAIL`，host 无 ascend 后端属预期 FAIL，恰好演示了两轨的分别）。
产品件腿预算 `timeout 900`（容器冷编译实测 4–7s，卡上首编留余量）。
容器侧注意：必须先 source `set_env.sh`（本镜像 bisheng 在 `cann-8.5.0/bin`，`card_run.sh` 里
`ccec_compiler/bin/bisheng` 那条 find 命不中 ⇒ `BISHENG_HOME` 留空但 PATH 有件即可）。

---

## 6. 已知限制 / 待裁决

1. **本窗无真机判决行**：真机 rel、`aicore` 侧行为、507015 是否复现，都留下窗（波次边界）。
2. **cpu-alias 轨不覆盖 ascend 发射层**：差异只在 dialect 的 storage/发射映射，由容器 `ASC-COMPILE-PASS`
   + 生成码体（`src~NB`）兜；真机判决才是终局。
3. **`DISC_MIN_SCATTER=0.3` 是本波实测定标的常量**（非来自上游尺子）：若下窗真机出现 0.3–0.5 区间的
   可疑距离，需要重新定标而不是直接放行。
4. **待裁决（不阻塞本任务）**：`compile_all_asc.py` 与 `run_numerics.py` 的产品件 case 现在形状族不同
   （前者一件一形状取编译判决，后者带尾块/朝向/复用判别族）——是否合并成单一判决清单入口，请上级定。
5. **观测到一次未复现的空输出**：容器里首轮 `for t in gemm_l1_intrinsic readout_intrinsic; do … | grep | head`
   两条 case 均无判决行；单独重定向跑（EXIT=0，判决行在）与复跑同形态循环（两行都出）均正常，冷编译实测
   仅 2–7s（远小于 `timeout 500`）⇒ 结论：非超时、非脚本形态问题，疑为容器内并发编译争用；**如实记录，未修**。

## 7. 产物清单

- 修改：`release/ascend/port910b/run_numerics.py`（+ 三条产品件 case、四case 改名、守卫与三档 track；`--only` 全表见 §1）
- 修改：`release/ascend/compile_all_asc.py`（九腿改走入口件、`--target`、判据前缀随 target、退出码）
- 修改：`release/ascend/port910b/card_run.sh`（overlay 落点修正 + 存在性断言 + [3] 产品件优先 + 自证钩子）
- 新建：`release/ascend/kernels/reconcile/Y_RESULT.md`（本件）
- **并行代理的改动混在同一 mtime 窗口**（本代理未触碰）：`attempts/X_probe_*.py`、`X_carve_hivmc.py`、
  以及 `kernels/letter_readout_asc.py`(23:43) / `S_RESULT.md` / `U_oracle.py` 等为 X/U/S 代理落笔时刻；
  本代理凭据全部产出于 00:1x–00:3x（即上述文件当前状态之后），复跑仍绿。
- 临时探针/补丁（**用完即弃，不参与判决，可删**）：`scratch/y_probe_prod.py`、`scratch/y_probe_prod2.py`、
  `scratch/y_patch1.py`、`scratch/y_patch2.py`、`scratch/y_guard_proof.py`（负例自证件，建议保留以便复跑 §0.2）、
  `scratch/fake_tlroot/`（card_run.sh [2] 段自证沙盒）
