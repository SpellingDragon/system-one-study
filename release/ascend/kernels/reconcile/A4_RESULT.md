# A4_RESULT — p2-13 执行代理 **A4**：`addln_prod` 产品件接入（P1-1x）

- **变更**：`teacher-p2-production-full` / 嵌套子变更 `p2-13-ascend-runtime`（openspec CLI 不索引孙层，按任务书文本直读）
- **孙任务**：修 C3 收口复核发现的错位——「判决件量的不是产品件」在 `linear/dw/readout` 三件已闭合（Y 的 `*_prod` 腿），**add_ln 仍是 attempts 形态**（`grep -c addln_prod` 全库=0）
- **本窗性质**：**host（无卡）+ 容器 `cann910b-h`（compat md5 `ca023c2e`、有 bisheng 无设备）**；**不占卡**，不含真机判决行（下窗产生）
- **改动件（白名单内）**：`release/ascend/port910b/run_numerics.py`（仅新增 `addln_prod` case + 存储层 shim + 说明）；`compile_all_asc.py` **零改动**（其产品腿清单由 `run_numerics.PROD_TARGETS` 自动派生，见 §④）
- **一字未动**：任何 `*_asc.py` / `*_kernel.py` / `ascend_env.py` / `autograd_asc.py` / `train_step.py` / `bench_card.py` / `card_run.sh` / `run_delta_card.py` / trunk / `attempts/D2_*` / `reconcile/{S3,V2,U2,M2,C2,C3}_*` / `runs/**` / **任何 tasks.md**；**未做 git 写**
- **一句话结论**：`addln_prod` 腿已上轨并**真接产品接口件**（`add_ln_kernel.forward → add_ln_asc.add_ln_asc_impl`）：host cpu-alias `Y-addln_prod-PROD-PASS rel_y=1.84e-05 rel_h=0.00e+00 unb_disc=7.36e-04 reuse=1/1`；容器 `Y-addln_prod-ASC-COMPILE-PASS key=ascend|add_ln[ascend]|d512e1e-05b16`；聚合产品腿 **3→4**；三件既有 `*_prod` 判决行**逐字未变**；host 全量 pytest **340 passed / 0 failed**。
- `run_numerics.py` md5 **d3b076d011d2d67b3790a6efc9b78f0c**（949 行）；`compile_all_asc.py` md5 **bd7a383665f54d35dc95bb90e41910f0**（**未改**）

---

## ① 完成情况（逐条附可复跑凭据）

| 子项 | 状态 | 凭据（可复跑命令 → 真实输出） |
|---|---|---|
| 接入 `addln_prod` case（照 `readout_prod`/`linear_prod` 形态，走 `add_ln_kernel` 入口件） | ✅ | `run_numerics.py` 新增 `ADDLN_SHAPES`/`ADDLN_REL_*` 常量、`_addln_cpu_alias_storage_shim()`、`prod_addln()`，并注册进 `PROD_TARGETS`（md5 `d3b076d0…`）；`--only bogus` 的可用表已含 `addln_prod` |
| 判据沿用本件既有阈值，**一字不改** | ✅ | `ADDLN_REL_H=1e-5 / ADDLN_REL_Y=3e-4 / ADDLN_DISC_UNB=2e-4 / ADDLN_EPS=1e-5`——与 §7 `num_addln_intrinsic` 的宿主机仿真口径同值；`grep -n "ADDLN_REL_H\|1e-5" run_numerics.py` 可核 |
| host cpu-alias 真跑 → PASS（给 rel 数值） | ✅ | `cd release && ./.venv/bin/python ascend/port910b/run_numerics.py --only addln_prod --track cpu-alias` → `Y-addln_prod-PROD-PASS rel_y=1.84e-05 rel_h=0.00e+00 unb_disc=7.36e-04 reuse=1/1 shapes=2 track=cpu-alias`（rc=0） |
| 容器 `cann910b-h` 编译判决 PASS、键含位宽段 | ✅ | `docker exec cann910b-h … python3 ascend/port910b/run_numerics.py --only addln_prod --track card --no-run` → `Y-addln_prod-ASC-COMPILE-PASS keys=1 key=ascend|add_ln[ascend]|d512e1e-05b16 src~3652B track=card(no-run)`（rc=0；键含 `d512` 宽度段） |
| 旧 attempts 形态不删 + 加 `[note]` | ✅ | `num_addln_intrinsic` docstring 增「判决地位已由 `addln_prod` 取代…results 不变、不删」；`DEPRECATED` 别名 `addln→addln_intrinsic` 保留。实证：`--only addln` → `[note] …弃用别名 ⇒ addln_intrinsic（attempts 内联正文，performance-track，非产品件）` |
| `compile_all_asc.py --target cpu` 仍 9/9 | ✅ | `./.venv/bin/python ascend/compile_all_asc.py --target cpu` → 九条 `ASC-* PASS` + `—— 汇总 target=cpu 接口腿 PASS=9 FAIL=0 —— 门禁失败数=0`（rc=0） |
| `--target ascend --products` 产品腿计数 3→4 | ✅ | 容器 `python3 ascend/compile_all_asc.py --target ascend --products` → `产品腿 PASS=4 FAIL=0`（`ASC-PROD-addln_prod PASS Y-addln_prod-ASC-COMPILE-PASS key=ascend\|add_ln[ascend]\|d512e1e-05b16`），接口 9/9、对照 8/8，rc=0 |
| host 全量 pytest 不回归 | ✅ | `cd release && ./.venv/bin/python -m pytest tests -q -m "not integration"` → **340 passed, 3 skipped, 3 deselected, 0 failed** |
| **回归红线**：三腿判决行逐字同 | ✅ | before/after 见 §② |

---

## ② 回归红线（before/after 逐字对照）

**before（改 `run_numerics.py` 之前）**
```
Y-linear_prod-PROD-PASS  rel=2.54e-07 dir_disc=1.37e+00 reuse=4/4 relu_neg_clipped=802 shapes=5 track=cpu-alias
Y-dw_prod-PROD-PASS      rel=2.21e-07 dir_disc=1.38e+00 reuse=4/4 shapes=4 track=cpu-alias
Y-readout_prod-PROD-PASS rel=0.00e+00 fwd=0.00e+00 bwd=0.00e+00 bitexact=True reuse=6/6 shapes=5 track=cpu-alias
```
**after（改后复跑，`--track cpu-alias`）**
```
Y-linear_prod-PROD-PASS  rel=2.54e-07 dir_disc=1.37e+00 reuse=4/4 relu_neg_clipped=802 shapes=5 track=cpu-alias
Y-dw_prod-PROD-PASS      rel=2.21e-07 dir_disc=1.38e+00 reuse=4/4 shapes=4 track=cpu-alias
Y-readout_prod-PROD-PASS rel=0.00e+00 fwd=0.00e+00 bwd=0.00e+00 bitexact=True reuse=6/6 shapes=5 track=cpu-alias
```
⇒ **逐字相同**。新增 case 未触碰 `_prod_setup`/`_route_key`/三件既有 case 的任何一行。

---

## ③ 负例自证（新腿真接产品件，不兜底旧 attempts）

`reconcile/A4_probe_neg.py`：把产品件 `add_ln_asc._plan_fwd` 桩成恒 `None`（入口件必落回 torch `_ln` 快路），
`addln_prod` **必须 FAIL**：

```
cd release && ./.venv/bin/python ascend/kernels/reconcile/A4_probe_neg.py
Y-addln_prod-PROD-FAIL AssertionError: 假绿守卫：compiled_keys 无 ascend|*add_ln* ⇒ 本趟走的是 torch eager，
  不是产品件（方言别名未生效 / plan 拒形状 / 后端不可用）；blockers=[]
NEG-EXIT-CODE=1（期望 1）        process_rc=1
```
⇒ plan 一坏就变 FAIL，说明新腿量的确是产品接口件、**不会**去找 §7 的旧 attempts 内联件兜底
（旧件 `addln_intrinsic` 是另一份能自编出键的正文，与新腿互不相干）。

---

## ④ 关键机制 / 偏离记录（如实申报）

### D1（**计划外的必要改动，白名单内**）：cpu-alias 轨加存储层 shim `alloc_shared`→`alloc_local`

- **现象**：`add_ln_asc_impl` 在 host 直连 cpu 方言（`cpu-alias` 原装手法）**编不过**：
  `InternalError: Check failed: (!dyn_shmem_size.defined()) is false: Only one dynamic shared memory allocation is allowed.`
- **机理**（探针 `A4_probe_addln_compile.py`，before/after 现成）：
  `add_ln_asc_impl` 是 **ascend-only 正文**，用 4 个 `T.alloc_shared`（scope=`shared.dyn`，
  `tilelang/language/allocate.py:49`）；c 后端的 `lower_device_kernel_launch`
  （`src/transform/lower_device_kernel_launch.cc:173`）只允许**一个** dynamic shared 分配。
  gemm/dw 的产品正文靠 `gemm_asc._alloc_tile` **按方言分叉存储层**（cpu→`alloc_local`）；
  readout 正文本身就用 `alloc_local`。**add_ln 件没有这层分叉**（它只为昇腾写）。
- **处置**：`prod_addln` 在 **cpu-alias 轨**额外装一层 shim，把该轨的 `tilelang.ascend.language`
  再包一层、`alloc_shared` 换名到 `alloc_local`——即替产品件做 `_alloc_tile` 同一件事。
  探针实测：原装 FAIL / shim OK `src~2941B` / readout 对照 OK。
- **边界（不夸大）**：只换**存储层**（UB vs local），**算式、形状、判据阈值一字不动**；cpu-alias 轨
  本就只判"无卡时能拿到的最强数值证据"，真 UB 发射层差异由 card 轨/容器编译判决负责。shim 只在
  `prod_addln` 内、只在 cpu-alias 轨安装，`_prod_setup().restore()` 一并还原。

### D2（**未做计划中"改 compile_all_asc.py 清单常量"**，因无需改）
任务书预见"聚合入口在 A2 手里，若其枚举需更新则改 `compile_all_asc.py` 的清单常量"。实读 `compile_all_asc.py:348`：
`_prod_names` 的 `all/prod` 分支取 `list(R.PROD_TARGETS)`——**产品腿清单是从 `run_numerics.PROD_TARGETS` 自动派生的**。
故把 `addln_prod` 注册进 `PROD_TARGETS` 后，聚合入口**零改动**即得 3→4。`compile_all_asc.py` md5 与改前一致（`bd7a3836…`）。

### D3（**键含位宽段的解释**）
`add_ln` 产品正文恒 fp32 内部计算、`out_dtype` 只是交货参数 ⇒ 其缓存键 `add_ln[ascend]|d{dim}e{eps}b{bm}`
**没有 dtype 段**；本任务的"位宽段"落到**宽度轴 `d{dim}`** 上（与 readout_prod 的 `d{dim}` 同类）。
容器实测键 = `ascend|add_ln[ascend]|d512e1e-05b16`（含 `d512`）。若上级口径的"位宽段"指 dtype 段，
则需改产品件 `_plan_fwd` 的 key（**禁改件**）——本代理按"不改产品件"如实报此点。

---

## ⑤ 产物清单

- **修改**：`release/ascend/port910b/run_numerics.py`
  - 模块 docstring：② 三条→四条产品件 case（加 `add_ln_kernel`）+ 新增 ③ 段说明 shim
  - 新增常量：`ADDLN_REL_H/ADDLN_REL_Y/ADDLN_DISC_UNB/ADDLN_EPS`、`ADDLN_SHAPES`
  - 新增函数：`_addln_cpu_alias_storage_shim()`、`prod_addln()`
  - `PROD_TARGETS` 加 `addln_prod`
  - `num_addln_intrinsic` docstring 加「判决地位已被 `addln_prod` 取代」note（正文/判据/返回值**未动**）
- **新建（本轮探针，可删；判决面在 `run_numerics.py`）**：
  - `release/ascend/kernels/reconcile/A4_probe_addln.py`（原型：复用 run_numerics 装配跑通 case）
  - `release/ascend/kernels/reconcile/A4_probe_addln_compile.py`（机理探针：原装 vs shim vs readout 对照）
  - `release/ascend/kernels/reconcile/A4_probe_shared.py`（旁证：c 后端多 `alloc_shared` 行为）
  - `release/ascend/kernels/reconcile/A4_probe_neg.py`（负例自证）
- **新建**：`release/ascend/kernels/reconcile/A4_RESULT.md`（本件）
- **未改复核**：`compile_all_asc.py`（md5 未变）；所有 `*_asc.py`/`*_kernel.py`/`ascend_env.py`/`autograd_asc.py`/`train_step.py`/`bench_card.py`/`card_run.sh`/`run_delta_card.py`/trunk/`attempts/D2_*`/`reconcile/{S3,V2,U2,M2,C2,C3}_*`/`runs/**`/**任何 tasks.md** 一字未动；无 git 写

---

## ⑥ 已知限制 / 待裁决

1. **本窗无真机判决行**：`Y-addln_prod-PROD-PASS` 出自 host cpu-alias（生产 ascend 正文 + cpu 后端真跑），
   容器只到 `ASC-COMPILE-PASS`；真机 rel 留卡窗（`python3 ascend/compile_all_asc.py --target ascend --products`
   在有卡时自动升级为真数值判决，`addln_prod` 已随 `PROD_TARGETS` 自动纳入）。
2. **cpu-alias 轨的存储层 shim（D1）是本波新增的自定机制**：若上级要求"cpu-alias 必须零改动产品正文/零 shim"，
   则 add_ln 在 host 只能走 `--track card`（host 无 ascend 编译面 ⇒ 不可用）⇒ 会退化为"host 无证据"。
   本代理按任务书"host cpu-alias 必须 PASS"取 shim 路线，并如实登记。
3. **`addln_prod` 只判前向**（r_h/r_y/unb_disc，照任务书点名的既有阈值）：反向 `ln_bwd` 的数值/编译判决
   不在本件（`verify_addln.py` 有编译判决；C3 §1f 的 gradcheck 卡上腿另立）。
4. **两档形状同 dim（512）⇒ reuse=1/1**：这是**有意**用于证"行数不进缓存键"的产品行为；若上级希望覆盖
   多 dim 的键隔离，可再加 dim 档（属形状扩展，非判据改动）。
