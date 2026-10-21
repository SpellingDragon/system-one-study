# run: 1005-p213-ascend-cpu-probe

- 域：p2-13-ascend-runtime（新增域，p2-05/06/11 三栈与 p2-10 推理的公共底座）
- 波次：**本地半场 R1+R2**（R3+ 全部等云端开卡，本份不含任何 NPU 实测数字）
- 日期：2026-10-05　执行：p2-13 apply agent　环境：`release/.venv` python 3.12.8 / torch 2.14.1 / **tilelang 0.1.15**（禁装新包）
- 硬约束落实：MPS 被一阶段占用 → **全部验证走 `target="cpu"`**；ascend target 只写码不实编；不 import TileKernels/scratch；不碰 sys1/decision/；不真碰 NPU；不 git commit。

- **假设**：TileLang 0.1.15 的昇腾方言在 CPU(c) 后端能等价表达同一套算式，因此"本地把七类件的语义与梯度口径钉死、云端只做编译与性能"这条路成立；且 P1 冻结接口（`sys1/kernels/*`）可直接作为本域件形与签名的尺子。
- **观察**：`pytest tests/test_ascend_gradcheck.py -k cpu -q` → **20 passed, 1 deselected, EXIT=0**；全量 `-q` → **20 passed, 1 skipped, EXIT=0**；`bash ascend/env_setup.sh --dry-run` → **EXIT=0**（9 步清单全打印，Linux 判定在本机正确给出 warn 分支）；`python ascend/selfcheck.py --allow-ascend-missing` → **EXIT=0，PASS=5 FAIL=0 SKIP=2 KNOWN-GAP=2**；`check_comments release/ascend` → **18 文件，✅ 注释门通过**。两处 KNOWN-GAP 是**本机无卡的本因**，原文：`Check failed: (reg != nullptr) is false: Operator tl.tileop.ascend_copy is not registered`、`RuntimeError: Cannot find the bisheng compiler.`；七件里六件 CPU 对拍误差 ≤1.4e-06，**GDN 反向按 plan 如实标 partial**（入口走 torch 混合栈，阻塞账里可见）。
- **结论**：p2-13 本地半场交付成立——八枚入口件（七类算子，linear 拆前向/dW 反向两枚）在 `target="cpu"` 下对 fp32 torch 参考全绿且**内核真发射**（用例断言编译产物键存在 + 阻塞账为空，回退冒充通过会被抓住）；ascend target 写法已按 0.1.15 源码探查定档并写进 `ascend_env.compile_kwargs()`，唯一 partial 是 GDN 反向（昇腾方言无先例，官方 GDN 反向件全是 CUDA 方言）；云端 C1 必须先解 `ascend_copy is not registered` 与 bisheng 缺失两关，且 `ASCEND_NPU_ARCH` 默认 `dav-3510` 与 910B 需要的 `dav-2201` 不一致——这条是 D6 回退决策的第一触发点。

---

## 1. ascend target 写法探查结论行（抄此段即可，勿再猜）

| # | 结论 | 凭据（可复跑定位） |
|---|---|---|
| 1 | 目标名就是字符串 `"ascend"`；规范化只认 `target.strip()=="ascend"`，另有简名 `"asc"`（`register_target_normalizer("asc", ...)`） | `.venv/.../tilelang/ascend/target.py:40-59` |
| 2 | 官方样例统一写法即 `tilelang.compile(prog, target="ascend", out_idx=-1)`；jit 装饰式写法也在用（`@tilelang.jit(out_idx=[], target="ascend")`） | `tilelang/examples/ascend/example_gemm.py:124`、`example_gemm_splitk.py:110`、`tilelang/profiler/msprof.py:13` |
| 3 | 后端注册：`BACKEND name="ascend"`，`target_kinds=("ascend",)`，pipeline/codegen/execution_backends 都挂在 ascend 模块下 → 方言 import 口径为 `import tilelang.ascend.language as T` | `tilelang/ascend/backend.py:52-59` |
| 4 | 昇腾执行后端候选 = `[tvm_ffi, cython]`，**tvm_ffi 排首位**（`execution_backend="auto"` 落它）。故本域昇腾侧**只传 target、不钉 execution_backend**；CPU 侧才必须三件套 `target/target_host/execution_backend="c"/"c"/"cython"` | `tilelang/ascend/execution_backend.py:5-24`、`tilelang/cache/__init__.py:32-34,69` |
| 5 | 产物缓存按后端分派：`context.module.name=="ascend"` 走 `AscendCython/AscendTVMFFIKernelCache`；磁盘目录 `ascend-binaries`（`.aibin`） | `tilelang/cache/__init__.py:16-34,69`、`tilelang/cache/ascend_binary_cache.py:15-19` |
| 6 | 设备在位判据：官方 `testing/_ascend_device_available()` 明确 **不能用 `tvm.device("ascend").exist`**（"Unknown optional runtime ascend"），**必须问 torch_npu**。与本域 `ascend_env._npu_present()`（`hasattr(torch,"npu") and torch.npu.is_available()`）同口径 | `tilelang/testing/__init__.py:77-96` |
| 7 | 编译器与架构位：bisheng 取 `--npu-arch`，缺省读 `ASCEND_NPU_ARCH`，**默认值 `dav-3510`（950 系）**；910B 需显式给 `dav-2201`（本域 env_setup 已 pin `NPU_ARCH_910B=dav-2201`）。查找链 `BISHENG_HOME / ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME` | `tilelang/contrib/bisheng.py:93`、`bisheng.py:52-70` |
| 8 | 昇腾侧先例清单（可直接抄的口径）：`examples/ascend/` 有 gemm / gemm_splitk / rmsnorm / vecadd / per_token_cast_to_fp8 / **flash_attention（含 `core_bwd.py`、`example_gqa_bwd.py` 反向先例）** / deepgemm；**没有任何 delta rule 件** | `ls examples/ascend/ examples/ascend/flash_attention/` |
| 9 | TileKernels 昇腾件里**有** `engram_gate_bwd_asc.py / moe_topk_gate_backward_asc.py / swiglu_backward_asc.py` 等反向先例，**独独没有 gated-delta 反向**；官方 GDN 前向/反向件（`examples/gdn/example_chunk_delta_bwd.py` 等）全是 CUDA 方言（`import tilelang.language as T` + `.cuda()` + cuda postproc） | `grep -rli "delta\|gdn" TileKernels/tile_kernels/`、`examples/gdn/example_chunk_delta_bwd.py:5-6,45` |
| 10 | 昇腾 L1 乘加只认 `transpose_A=False / transpose_B=True`（故权重按 `(N,K)` 存）；DMA 搬运不得顺带转类型（`tilelang/ascend/analysis/vf_checker.py`）——本域所有件出口一律 fp32，降位放入口层 | 前段读 `tilelang-ascend` SKILL + 手册；已在件内 docstring 落档 |

**本域开关落点**：`ascend_env.normalize_target()` 白名单（`ascend/asc/npu→ascend`、`cpu/c/host→cpu`，其余 `raise ValueError`）+ `ascend_env.compile_kwargs(target)` + `ascend_env.dialect(target)` 三方言工具箱；件内按 `xxx_asc_impl / xxx_cpu_impl` 两份正文、同一 `plan()/run()` 口径，`--target ascend|cpu` 一路透传。缓存键经 `make_key` 强制带 `cpu|`/`ascend|` 前缀，防两种后端模具在缓存里互顶。

## 2. 七类件状态表（R2；全绿/partial 及原因）

| # | 类别 | 方言件 | 入口壳（对外名，与 P1 同名同签名） | 状态 | CPU 对拍凭据（真实输出） |
|---|---|---|---|---|---|
| 1 | linear 前向 | `gemm_asc.py`（`gemm_cpu_impl`/`gemm_asc_impl`） | `gemm_kernel.forward(A,W,bias,act,out_dtype,target)` | **全绿** | `o21: fwd 0.0`；pytest `test_gemm_cpu_matches_torch_ref` + `test_gemm_cpu_reuses_single_compilation`（换批 `compile_count()` 不变，M 为动态维） |
| 2 | linear dW 反向 | `gemm_bwd_dw_asc.py` | `gemm_bwd_dw_kernel.backward(dY,A,*,out_dtype,target)` | **全绿** | `o22: dw 0.0 nc 1 blk {}`；`test_gemm_bwd_dw_cpu_matches_torch_ruler` |
| 3 | rope 负角反向 | `rope_asc.py`（`sign=+1/-1` 两份键） | `rope_kernel.forward(qkv,cos,sin,slots,target)` / `backward(...)` | **全绿** | `o44: fwd_err 1.192e-07 / bwd_err 2.384e-07 / entry_err 2.384e-07 / eager_same_obj True / nc 2 blk {} keys ['cpu\|rope[cpu]\|h3d8s2g-1','cpu\|rope[cpu]\|h3d8s2g1']`；往返恒等用例 roundtrip 0.0 |
| 4 | add_ln 闭式反向 | `add_ln_asc.py`（`add_ln_impl` + `ln_bwd_impl`） | `add_ln_kernel.forward(x,residual,weight,bias,eps,out_dtype,target)` / `backward(...)` | **全绿** | `o29: h_err 0.0 y_err 4.77e-07 / dx 7.15e-07 dg 1.43e-06 db 9.54e-07 / nc 2 blk {}`（闭式反向对 autograd 交叉） |
| 5 | letter_readout 读出 | `letter_readout_asc.py`（`gather_impl` + `scatter_add_impl`） | `letter_readout_kernel.forward(rows,ids,out_dtype,target)` / `backward(row_count,ids,dy,...)` | **全绿** | `o34: gather_err 0.0 scatter_err 0.0 autograd_err 0.0 nc 2 blk {}`；重复点名按 `got[3]==dy[0]+dy[3]` 真累加（不覆盖） |
| 6 | attn_sw 滑窗 | `attn_sw_asc.py` | `attn_sw_kernel.forward(q,k,v,window,scale,out_dtype,target)` / `forward_weights(...)` | **全绿** | `o37: kernel_err 3.58e-07 torch_ref_err 3.58e-07（与 P1 torch_ref 交叉）rowsum 1.19e-07 fullcausal 4.77e-07 nc 2 blk {}`；不可见位权重逐位 0（`masked_select(~allow)`） |
| 7 | **GDN（最重件）** | `gdn_asc.py` | `gdn_kernel.forward(q,k,v,g,beta,out_dtype,target)` / `backward(...)` | 前向 **全绿**；反向 **partial（如实标）** | `o39: gdn_fwd_err 1.788e-07 / gdn_batch_err 1.788e-07 shape (1,2,24,8) / nc 1 blk {'gdn_backward[cpu]': '本波未内核化：昇腾方言无 Gated DeltaRule 反向先例（见 gdn_asc 模块 docstring）'} keys ['cpu\|gdn[cpu]\|h2k8v8']`；`gdn_bwd_status partial`、`gdn_bwd_errs [0.0,0.0,0.0,0.0,0.0]` |
| 8 | LoRA 注入（组合件） | `lora_asc.py`（**无自有方言正文**，由 #1/#2 组合） | `lora_kernel.apply(x,A,B,scaling,base,out_dtype,target)` / `backward(...)` / `merge(...)` | **全绿** | `o39: lora_fwd_err 0.0 / lora_base_err 0.0 / lora_bwd_errs 1e-07 5e-07 0.0 / lora_merge_err 2.98e-08 / nc 1 blk {}`；`merge` 与 `apply` 两条路互校（8.94e-08） |

- 七类 = 八枚入口（linear 拆前向 + dW 反向两枚，与 P1 的两个文件一一对应）。
- 每件都验了**"内核真发射"**而非只验数值：`_assert_kernel_ran(fragment)` 断言 `ascend_env.compiled_keys()` 里存在 `cpu\|<件名>[cpu]\|...` 且阻塞账无该项——**回退冒充通过会被直接判失败**（这条用例在首跑时确实抓住过 5 条假绿，见 §5）。
- 支持窗外形状另有一条专测 `test_gemm_cpu_falls_back_outside_block_window`：`N/K` 不被 `BLOCK_LADDER=(64,32,16)` 任一块宽整除时，`plan()` 返回 None → 入口落 torch，**数值仍对且不留编译产物**（把"回退是设计"钉成可执行文档）。

## 3. 验收命令与真实输出（五条，均可复跑）

```bash
cd $REPO_ROOT/release
.venv/bin/python -m pytest tests/test_ascend_gradcheck.py -k cpu -q   # 20 passed, 1 deselected, 1 warning in 7.78s → EXIT=0   (o49)
.venv/bin/python -m pytest tests/test_ascend_gradcheck.py -q          # 20 passed, 1 skipped, 1 warning in 2.28s → EXIT=0     (o50)
bash ascend/env_setup.sh --dry-run                                    # 9 步 pin 清单打印完 → EXIT=0                              (o51)
.venv/bin/python ascend/selfcheck.py --allow-ascend-missing           # PASS=5 FAIL=0 SKIP=2 KNOWN-GAP=2 → EXIT=0               (o52)
cd .. && release/.venv/bin/python scratch/tools/check_comments.py release/ascend   # 扫描 18 个文件 ✅ 注释门通过
release/.venv/bin/python runs/1005-p213-ascend-cpu-probe/lint_substitute.py        # ✅ AST 体检通过 → EXIT=0（ruff 缺失的替代凭据）
```

用例名规则（供 `-k cpu` 精确命中）：19 条名字含 `cpu`；唯一 ascend-only 用例取名 `test_attn_sw_ascend_target_semantics_on_npu`（**不含 cpu**，故 `-k cpu` 自动排除、开卡后 `-q` 会因无卡 skip），带 `@pytest.mark.npu`。

## 4. TileLang 0.1.15 坑清单（写给 R4 云端实施者，逐条实测）

1. **方言件禁 `from __future__ import annotations`**：PEP 563 把注解变字符串后，**只在注解里出现**的形状常量（`m/n/k`）没有闭包单元，`eager/builder.py: get_func_nonlocals` 拿不到 → 编译期 `NameError: name 'n' is not defined`。宿主件（入口壳/`ascend_env`/`selfcheck`）带它无害——探针形参只引用字面量与体内外都出现的 `T`；本机 `backend_available('cpu')=True` 即实测佐证（`o48`）。本域已把这条做成可执行守卫（`lint_substitute.py` 的坑①检查，负控验证会响）。
2. **禁把 `tilelang.jit` 返回值当执行**：件形是 lazy 工厂，`jit(...)` 只回 Kernel 对象不真跑（出口保持全零）；其 call-form 缓存拿张量做 `==` 比较上次调用（`jit/__init__.py: _CallFormCache._matches_last`），换新张量直接 `RuntimeError`。正确口径：`tilelang.compile(spec["impl"](tensors, **kwargs), out_idx=[], **ascend_env.compile_kwargs(target))` 后 `kernel(tensors...)`。
3. **CPU(c) 后端网格上界必须是编译期常量**：把动态 `m` 写进 `T.Kernel` 网格会**一次都不发射**（实测 C 全零，误差恰好等于 `|A@Wᵀ|` 最大值——最阴的假绿）。口径：列块静态网格 + 行块 `T.serial(ceildiv(m,bm))` / 昇腾侧固定 `NUM_BLOCKS` + `T.Pipelined` 循环界。
4. `T.reduce_sum` 第一参必须是 Buffer（不是表达式）。
5. `torch.autograd.grad(out, leaves, g)` **只返回梯度、不写 `leaf.grad`**：读 `t.grad` 恒为 None（`gdn_asc.backward` 曾因此 `AttributeError: 'NoneType' object has no attribute 'contiguous'`，已修并加口径注释）。
6. 昇腾 lowering 两堵本机墙（**云端首验目标，非本域缺陷**）：`Operator tl.tileop.ascend_copy is not registered`（方言 op 注册缺，说明 import 成功但 lowering 未起）；`Cannot find the bisheng compiler`（缺 AscendC 编译器/未设 `BISHENG_HOME`）。
7. `ASCEND_NPU_ARCH` 默认 `dav-3510`（950 系）；**910B 要显式 `dav-2201`**（env_setup 已 pin）。
8. tilelang 有**磁盘编译缓存**：重复跑同一件从 17.6s 降到 2.3s。所以"变快了"**不能**当"内核发射了"的证据；验收凭据用 `compile_count()`（进程内真实编译次数）与 `compiled_keys()`。

## 5. 事故与恢复记录（必须入库的负结果）

- **自伤**：给 21 处嵌套 `@T.prim_func` 补 docstring 的脚本漏了三引号 → `SyntaxError`；随后的"修复"脚本删除条件里含了 `白话` 二字，**贪婪匹配**把 `gemm_asc.py` 从 192 行删到 153 行，吞掉外层 docstring 收尾、方言 import、`m/nb`、`@T.prim_func`+def 签名、`plan()` 函数体与整个 `run()`。
- **恢复**：用受损前的 `__pycache__/gemm_asc.cpython-312.pyc`（时间戳 14:30 < 受损 15:51）里原样保存的 docstring 常量 + `dis` 指令流行号反推代码体，重建回 **192 行、语法 OK**，再以 `-k cpu` 全量复跑证明行为未变。凭据链：`dump_pyc.py / dump_mod.py / dump_plan.py / restore_gemm.py / restore_gemm2.py`（本目录）。
- **教训**：① 对产码文件做"文本手术"必须 AST 定位 + 正确三引号 + **自下而上**插入；② 删除条件绝不能含正常文案词（`白话`）；③ pyc 是可用的最后凭据。
- **假绿抓捕实录**：pytest 首跑 5 failed `未见 gemm[ 的编译产物，实际键 = []` —— 根因是我挑的被测形状（n=10/r=4）落在 `BLOCK_LADDER` 支持窗外，`plan()` 如实返回 None 走回退，数值过而内核未发射。处置：形状改 16/32（进窗），并**追加窗外回退专测**把这条口径钉死。
- 注释门曾报 ~24 条违例（R2 对所有非下划线、体 >4 行的函数生效，**含嵌套 prim_func**）；另踩"写法必须是 `白话：` 紧跟冒号"——`白话（CPU 件）：` 不被正则识别，已全局改成 `白话：CPU 件这一份，` / `白话：昇腾件这一份，`（17 处）。

## 6. 遗留 / 云端 C1 待确认项（R3+ 开卡即办）

1. `ascend_copy is not registered` 是否随 CANN 9.2.0 + `-DUSE_ASCEND=ON` 的 TileLang 源码构建消解（**最大单点**；失败即按 D6 走 torch_npu 底座，算子故事降级）。
2. 910B 的 `--npu-arch` 真值与 `bisheng` 版本配套（pin 表里的 `dav-2201` 需按 CANN 支持表复核一次）。
3. **GDN 反向内核化**：昇腾无先例，本域按 plan 标 partial 并记阻塞账；若要补，候选路径是官方 CUDA 反向件（`examples/gdn/example_chunk_delta_bwd.py`，chunk-delta + wy-fast 两段式）语义移植，或 TileKernels `engram_gate_bwd_asc.py` 的门控反向范式借镜。**不硬凑**。
4. `ln_bwd` 的按列汇总本域用 `atomic_add`；昇腾已有先例是 partial + 两段式同步（`TileKernels` 列梯度件）。**开卡后二选一确认**，若 atomic 在 910B 上不可用则改两段式（件内已留 `plan()` 形状判定口，改动不动接口）。
5. bf16/fp16 容差口径：`selfcheck` C1/8 实测 出口降位 fp16=2.98e-02 / bf16=2.17e-01、中间降位 bf16=2.83e-01 → 本栈 fp16/bf16 对拍容差定 **2e-2**；R4 在真卡上复核。
6. TileKernels 后端自动切换在 910B 是否生效（`selfcheck` C1/6 本地只核对了判据一致性：`/dev/davinci_manager` 与 `torch.npu.is_available()` 同为 False，且**未设 `TILEKERNELS_ROOT`、不 import 外部件**）。
7. `ascend/kernels/*.py` 昇腾正文**本地从未实编**——任何"昇腾侧也绿"的表述都必须等 C2 的 `pytest tests/test_ascend_gradcheck.py -q`（NPU 环境）exit 0 才能写。

## 7. 环境缺口（如实记录）

- **ruff / pyflakes 本机不可用**：`which ruff` → 无；`.venv/bin/python -m ruff` → `No module named ruff`；`pyflakes` 同样缺失；conda base 亦无。任务书禁 pip install → **未安装**，改用 `runs/1005-p213-ascend-cpu-probe/lint_substitute.py`（语法 + F401 等价 + F811 等价 + 坑①守卫，四条负控已验证会响）作替代凭据。件内已有的 `# noqa: F401/F841` 标注是为将来真上 ruff 预留。
- 本目录内 `_selftest` 用后即焚的 `lint_selftest/`、`st.log` 为体检负控件，属凭据，保留。

## 8. 收尾波（本次会话最后一段）动作与三条新发现

| 动作 | 结果 / 凭据 |
|---|---|
| 复跑四条验收（`ascend_env.py` 又被动过，必须复验） | `-k cpu` → **20 passed, 1 deselected, 12.85s, EXIT=0**（`o53`）；全量 → 20 passed 1 skipped（`o50`）；dry-run EXIT=0（`o51`）；selfcheck **PASS=5 FAIL=0 SKIP=2 KNOWN-GAP=2**，EXIT=0（`o52`） |
| `ascend/selfcheck.py` docstring 手术后复验 | 探针 docstring 会被 eager builder 当语句 `__tb.eval` 走一遍，实测无害（`o52` C1/9 `max_err=0.00e+00`） |
| ruff 缺失 → 自建 AST 体检并做负控 | `lint_substitute.py`：真实件 EXIT=0，负控件（`lint_selftest/`）两条都响 → 守卫有效 |
| 修 docstring 续行缩进（仅 `ascend_env.py` 2 处 4 行 + 2 行尾随空格） | AST 定位 + **文案逐条比对一致**断言；改后 lint/gate 双复绿 |
| 清散件 | `release/` 根的 7 个本域临时探针（`_wtest_*.py`、`probe4.py`、`scratch_probe_*.py`）挪进 `strays_from_release_root/`（**只挪不删**）；非本域的 `_patch_rb.py/_fix_conclude.py` 未碰 |
| 勾任务 | 二级 `p2-13-ascend-runtime/tasks.md`：**R1、R2 已勾**（2/7），一级 tasks.md 未碰 |

三条新发现（都进坑清单，别的域也会踩）：

1. **CPython future 标志位不能写死**：3.12 实测 `annotations` 的位是 `0x1000000`（2²⁴）。我第一版写死 `0x10000` → **守卫静默空转**（永远判"没有 future-import"）。正确写法 `__future__.annotations.compiler_flag`。
2. **`compile()` 默认继承调用方的 `__future__` 设置**：体检脚本自己带 `from __future__ import annotations`，导致被扫的**每个**文件都判成"带 future-import"（假阳一片）。必须 `dont_inherit=True`。
3. **算"注解里的自由名"时不能整棵 `ast.walk(fn)`**：注解本身就是 `fn` 子树的一部分，整体走一遍会让 `k` 自己给自己作证，负控实测正是这样漏掉的；只遍历 `fn.body` 才对。

- 顺带纠一条我自己写粗的检索：`grep -l "from __future__ import annotations" ascend/kernels/*_asc.py` 命中 `gemm_asc.py`，其实是**模块 docstring 里那句"本文件不写 `from __future__ import annotations`"**被文本匹配到了；用 `grep -n "^from __future__"` 与 `co_flags` 双重复查后确认——**九个方言件都没有 future-import**，坑①红线仍然成立。教训：文本 grep 不能当结构判定用。

结论（编排者补录）：本地半场 R1+R2 达成——cpu 对拍 20 passed（GDN 反向 partial 按 spec 例外条），R3–R7 待开卡；细节见本 notes 状态表。
