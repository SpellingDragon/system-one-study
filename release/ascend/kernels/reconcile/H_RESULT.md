# H_RESULT.md — 代理 H / p2-13 P1-4 接口 home 910B 编译合流（add_ln + letter_readout）

**判决环境**：容器 `cann910b-h`（CANN 8.5.0 aarch64 + patched pip tilelang 0.1.15，§12 模板 +
GAP-B compat + patch_bisheng 预置；`ASCEND_NPU_ARCH=dav-2201`，无卡、只判编译）。
**结论一句话**：两件接口 home 的 ascend 路径已从 SimtVF 形态重写为 910B 可编形态，
**前反向共 4 个键全部 target=ascend 真编译 PASS**，target=cpu 语义路径不回归。

## ① 完成情况（逐条附凭据）

- [x] `add_ln_asc.py` ascend 路径重写（`add_ln_asc_impl` + `ln_bwd_asc_impl`）
  验证：`TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_addln.py` →
  ```
  keys=['ascend|add_ln[ascend]|d64e1e-05b16', 'ascend|ln_bwd[ascend]|d64e1e-05b16']
  .o~4697B simt_in_src=False rsqrtf_in_src=True gm2ub_in_src=True
  H-ADDLN-ASC-COMPILE-PASS
  ```
- [x] `letter_readout_asc.py` ascend 路径重写（`gather_asc_impl` + `scatter_add_asc_impl`）
  验证：`... python3 ascend/kernels/reconcile/verify_letter_readout.py` →
  ```
  keys=['ascend|gather[ascend]|d64', 'ascend|scatter_add[ascend]|d64']
  .o~1026B simt_or_atomic_in_src=False bounds_guard_in_src=True
  H-READOUT-ASC-COMPILE-PASS
  ```
- [x] target=cpu 语义不回归（两件的 cpu 正文**未改动**，直编直跑对拍）
  验证：`... verify_addln.py --cpu` → `fwd_ok=True bwd_ok=True entry_ok=True` +
  `H-ADDLN-CPU-SEMANTIC-PASS`；`... verify_letter_readout.py --cpu` →
  `gather_ok=True scatter_ok=True entry_f=True entry_b=True` + `H-READOUT-CPU-SEMANTIC-PASS`
- [x] 写后三连：`wc -l` add_ln_asc.py=471、letter_readout_asc.py=280；容器 `py_compile` OK；
  `grep 'SimtVF|T.Parallel|Persistent|atomic|annotate_buffer_versions|VEC_THREADS|NUM_STAGES|NUM_BLOCKS'`
  两文件 **0 命中**
- [x] 形态出处逐条锚定：动态网格+动态行（b_addln `--dyn` PASS）、UB 装填+标量三扫（b_addln 本体）、
  rsqrt=§11 `rsqrtf` shim（生成码含 rsqrtf=True）、按行号散读+codegen 界守卫（b_readout
  gen_readout.asc）、纯标量 store（ScalarDcacheBypass 取证）

## ② 错误与阻塞（含已解决的）

1. **设备门挡死编译路径（折回 compile_all_asc.py 必读）**：`ascend_env.active_backend("ascend",
   cpu张量.device)` 因 `want="npu" != "cpu"` 直接判 `torch_eager`，`forward/backward(target="ascend")`
   **根本走不到 tilelang.compile**——本 harness 只 patch `backend_available/_npu_present` 不够。
   → 处置：**已绕行**（verify_*.py 额外 patch `active_backend → TILELANG`，判据仍用
   `compiled_keys()` 含 `ascend|` 防假绿）。**编排者折回 `compile_all_asc.py` 时须补这层 patch
   （或 npu 元设备假体），否则这两件合流后仍会报 `PRECOMPILE-ERR 未达 tilelang.compile`。**
2. **容器内 cpu 后端自探测探针既有坏**：`ascend_env._compile_probe("cpu")` 在本容器
   （py3.10 + cython 适配）抛 `TypeError: Forward references must evaluate to types. Got buffer.`
   @ ascend_env.py:142 → `cpu:probe` blocker → 入口 target=cpu 全部落 torch_eager。
   → 处置：**绕行 + 上报**（ascend_env 属禁改件；判决改走"直编直跑 cpu 正文"——与 `_run` 同式
   的 builder + target="c" 三件套，day-1 对拍同一条编译链，PASS 见 ①；Mac 宿主机因
   ~/miniconda3 tilelang editable 指向缺失的 `tilelang/build/lib` 现也不可直用）。
3. **原子加无落点（本波形态选择的根因之一，非运行错误）**：`T.atomic_add` 在语言层存在
   （hasattr=True），但 codegen 发射的 `asc_atomic_add`（codegen_ascend.cc:2502-2531）在
   tilelang 模板与 CANN 8.5 aarch64 头文件内 **grep 0 命中** → 910B 无反向多块原子载体。
   → 处置：**已按接口件自陈的降级路线改单块串行**（见 ③-1），未造 compat 假符号。

## ③ 疑惑点与自行决策（凡"没问但自己定了"的事）

1. **反向落点改单块串行**（`ln_bwd_asc_impl`→`T.Kernel(1)` 串行 RMW 累计 Dg/Db；
   `scatter_add_asc_impl`→`T.Kernel(1)` 逐格"读—加—落"）：多块原子在 910B 无符号落点（②-3），
   串行与各自 cpu 件**逐字同式**、语义零变更（入口清零口径不变）。代价：反向占一个向量核。
   升级多块的两条路线（补 atomic compat / 偏量表两段归约+跨核事件）都需卡上证据，**留给开卡窗**；
   若编排者要求本波就投原子 shim，需裁决（本代理拒绝无落点造符号）。
2. **gather 不走 UB 批量搬运**，改逐格 GM 散读+标量 store：b_readout_910b 实证形态，且绕开
   GAP-B `asc_copy_gm2ub_align` 的 32B 粒度静默截断（dim 非 8 倍数时行字节非 32 整倍）。
   新增模块常量 `GATHER_ROWS_PER_BLOCK = 8` 顶替废弃的 NUM_BLOCKS/VEC_THREADS/NUM_STAGES。
3. **前向动态网格** `T.Kernel(T.ceildiv(rows, bm))`（b_addln `--dyn` 实证）；废弃
   `T.Persistent`/`T.annotate_buffer_versions` 流水形态（910B 无实证配置，且 SIMT 族整族出局）。
4. **尾块读越界待卡裁**：写侧已 `if row < rows` 标量守卫；`T.copy` 整块装填在 rows 不被 bm
   整除时**读**越界（b_addln --dyn 同形态编过；DMA 读越界的运行时行为本地不可判）。判决形状
   (16,64)/bm=16 无尾块。与 attempts/B 已知项同批，开卡数值窗一并裁决（R20）。
5. **接口 dtype 全 fp32 保留**：b_addln 实证含 fp16 激活档，但接口入口层本就 `.float()` 升位，
   合流跟随接口口径（fp32 标量面 = b_addln RDT=float32 档，同样实证可编）。bf16 面未触碰。
6. **未落 `compat_patch_H.h`**：本波选定形态零新增 compat 符号（rsqrtf/GAP-B 搬运件均已在
   容器 compat 内）；无需求不造占位文件。若后续投原子面再按 GAP_H 锚上报。

## ④ 偏离记录（与任务书/计划不一致之处）

| 计划原文 | 实际 | 理由 |
|---|---|---|
| 重写 `*_ascend_impl` | 实际函数名 `add_ln_asc_impl`/`ln_bwd_asc_impl`/`gather_asc_impl`/`scatter_add_asc_impl` | 以文件现状为准（任务书口述名），接口 `forward/backward/_plan/_run` 签名与 out_idx 语义未动 |
| "把 b_readout_910b 的 body 逻辑搬进接口" | 搬其**编译形态**（散读/动态下标/标量 store），不搬算式 | b_readout 语义=scores 投影（含点积），接口 readout 语义=纯 gather/scatter-add；张量 I/O 是契约，不可并 |
| 反向"原子加不稳就换两段式"（原件注释） | 换的是**单块串行**而非两段归约 | 两段式需跨核事件编排（asc_sync 组合在搬运件之外未经实证）；单块串行是零新风险的最短合流路，语义与 cpu 同式 |
| SimtVF 件判据只 forward | 前反向 4 键全判 | 反向同样含 SimtVF/atomic，留半个 SimtVF 件=上卡即崩 |

## ⑤ 产物清单与复现命令

**新增**：`ascend/kernels/reconcile/H_RESULT.md`、`reconcile/verify_addln.py`、`reconcile/verify_letter_readout.py`
**修改**：`ascend/kernels/add_ln_asc.py`（ascend 两 impl 重写+模块 docstring/常量同步；cpu 件逐字未动）、
`ascend/kernels/letter_readout_asc.py`（同）
**禁改件零触碰**：compile_all_asc.py / ascend_env.py / *_kernel.py / trunk / 其他 op 的 *_asc.py / sys1/（git status 自证见下）

真实可用的驱动形态（供 compile_all_asc.py 折回，均需 ②-1 的 active_backend 补丁）：
- `add_ln_asc.forward(x=(16,64)fp16, residual=(16,64)fp16, weight=(64,)fp32, bias=(64,)fp32, target="ascend")`
  → key `ascend|add_ln[ascend]|d{dim}e{eps}b{bm}`（bm=min(16, 8192//dim)）
- `add_ln_asc.backward(h=(16,64)fp32, weight=(64,)fp32, dy=(16,64)fp32, target="ascend")`
  → key `ascend|ln_bwd[ascend]|...`
- `letter_readout_asc.forward(rows=(16,64)fp32, ids=int64(4,)→int32, target="ascend")` → key `ascend|gather[ascend]|d64`
- `letter_readout_asc.backward(row_count=16, ids=int64(4,), dy=(4,64)fp32, target="ascend")` → key `ascend|scatter_add[ascend]|d64`

复现（每条都是全量判决，一次性缓存目录防 AscendBinaryCache 假 PASS）：
```bash
docker exec cann910b-h bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; cd /work && \
  TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_addln.py'
# 同法跑 verify_letter_readout.py；加 --cpu 跑语义不回归判决
```

**遗留给开卡窗（不在本波判据）**：四路数值 rel（fwd y/h、bwd dx/dg/db、gather、scatter）；
尾块 DMA 读越界行为；反向单核吞吐是否需升级多块；dim 非 8 倍数形状在 UB 搬运面的实测。
