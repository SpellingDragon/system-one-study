# C2 — P1-1m 产品件 dtype 矩阵（fp32/fp16 本地闭环 + bf16 缺件清单）

> 命题：**fp16 是生产位宽**（归约 fp32、出口不降位，见 `ascend/autograd_asc.py:128-134` `_work_dtype`
> 与 `:144` "fp16/fp32 走内核"），但接口 home 的标量件此前只在 fp32 闭环过。本窗把 8 行 × {fp32, fp16}
> 全部真编真跑对拍，并逐件实测 bf16 轨，把缺件如实列成清单（§8）。

## 0. 结论摘要

| 档 | 改前 | 改后 | 说明 |
|---|---|---|---|
| **fp32** | 9/9 PASS | **9/9 PASS**，逐格 rel 与改前**逐字同值** | 未回归（§3） |
| **fp16** | 6/9（linear/dw 发射期崩、rope NOKEY 静默落 eager） | **9/9 PASS** | 三件就地参数化（§4） |
| **bf16** | 6/9（三件崩） | **6/9 host 物化 PASS + 3/9 UNSUPPORTED（plan 期显式拒）** | 缺件清单见 §8；**不作支持宣称** |
| 卡面（910B target=ascend 编译） | — | **A=7/7 COMPILE-PASS**，生成码签名直证 `__gm__ half*` | §7 |
| 位宽缓存键隔离（同进程双档） | — | **9/9 格双档闭环** | §6 |
| host 全量回归 | 340 passed | **340 passed**（+3 skipped, 3 deselected） | `pytest tests -q -m "not integration"` |
| 独立工装旁证 | — | `verify_gemm.py --cpu` **PASS**（max_abs_diff=0）；`verify_rope.py --golden` 8 行 PASS | §3.1（cpu 语义路与循环结构口径均未动） |

## 1. 两轨口径（谁判什么）

| 轨 | 判什么 | 手段 | 凭据文件 |
|---|---|---|---|
| **host cpu-alias 数值轨** | 路由真不真、数值对不对、位宽守卫拒不拒 | `tilelang.ascend.language` 两处一起指向 cpu 方言 + `compile_kwargs` 翻 c 后端；递进去的 target 仍是 `"ascend"` ⇒ 路由/键名/正文全是产品件 | `reconcile/C2_probe_dtype_matrix.py`（配方源自 `run_numerics._prod_setup` 与 `S_cpu_oracle.py` 的 body 生成器 + `dialect(target)` 注入） |
| **容器 910B 编译轨** | 真 ascend 方言下位宽参数化后的正文编不编得出产物 | 绕设备门三处 + `tilelang.compile` 换"真编译后发射 no-op"替身（无设备 ⇒ 本轨**不判数值**） | `reconcile/C2_probe_card_fp16.py`（桩形态同 `verify_dw.py`） |

工装垫片两处（只补 cpu 方言缺失的**载体层**，不碰算式、不碰被测件）：`T.Vector()`→`nullcontext`；
`T.alloc_shared`→`T.alloc_local`（c 后端不允多个动态 shared 分配；口径同 `gemm_asc.py:120-128`
`_alloc_tile` 既有分叉"分存储层归属、不分算式"）。可行性凭据 `reconcile/C2_probe_alias_shim.py`。

**假绿守卫（load-bearing）**：入口件在 plan 拒形状/编译失败时会静默落 torch eager 并交出看起来对的数，
故每格**先断言 `ascend_env.compiled_keys()` 含 `ascend|<件>[ascend]` 再判数**；无键＝内核没跑＝不算 PASS。
**缓存纪律**：每次判决换新 `TILELANG_CACHE_DIR=$(mktemp -d)`（防陈旧 PASS）。

## 2. 主矩阵（8 行 × fp32/fp16，三态 + rel + 键）

判据沿用各件既有门限（fp32 列=原文阈值；fp16 列=2 ulp(fp16)≈1e-3，纯搬运件仍为逐位同）：
`verify_rope.py:214` max_abs<1e-5、`verify_attn_sw.py:223` max_abs<1e-4、`verify_gdn.py:227/239`
rtol=1e-4/atol=1e-5、`run_numerics.py:282-283` r_h<1e-5/r_y<3e-4、`S_cpu_oracle.py` REL_TOL=2e-5、
`verify_letter_readout.py` atol=1e-6。

| 行 | fp32 | fp16 | 内核位宽形态 | 键（fp16 档） |
|---|---|---|---|---|
| linear | **PASS** rel=1.387e-07 (<2e-5) | **PASS** rel=0.000e+00 (<1e-3) | **fp16 原生**：A/W/C 声明 fp16，UB 累加面恒 fp32 | `ascend\|gemm[ascend]\|n64k32b16x64x32anonedfloat16` |
| dW | **PASS** rel=2.258e-07 (<2e-5) | **PASS** rel=0.000e+00 (<1e-3) | **fp16 原生入 / fp32 出口**：dY/A 声明 fp16，`GW` 恒 fp32（归约不降位） | `ascend\|gemm_dw[ascend]\|n64k32tc16b64x32dfloat16` |
| readout | **PASS** rel=0.000e+00 逐位同 | **PASS** rel=0.000e+00 逐位同 | 入口升宽（`.float()` 物化） | `ascend\|gather[ascend]\|d32` |
| add_ln | **PASS** r_h=0.00e+00(<1e-5) r_y=2.62e-07(<3e-4) r_unb=7.90e-03 | **PASS** r_h=0.00e+00 r_y=6.54e-08 r_unb=7.56e-03 | 入口升宽 | `ascend\|add_ln[ascend]\|d64e1e-05b16` + `ascend\|ln_bwd[ascend]\|d64e1e-05b16` |
| gdn fwd | **PASS** rel=1.298e-07 (<1e-4) | **PASS** rel=1.661e-05 (<1e-4) | 入口升宽 | `ascend\|gdn[ascend]\|h2k16v16\|c8ubh` |
| gdn bwd | **PASS** dq1.3/dk2.0/dv1.2/dg1.7/dbeta1.8 e-07 | **PASS** dq1.3/dk2.0/dv1.2/dg1.9 e-07, dbeta9.2e-08 | 入口升宽；**五梯出口恒 fp32**（探针已 `assert a.dtype==torch.float32`，接口 fp16 入亦 fp32 梯度出）| `ascend\|gdnb[ascend]\|h2k16v16\|c2ubz` |
| rope | **PASS** rel=6.194e-08 (<1e-5) 槽位2逐位不动=True | **PASS** rel=0.000e+00 (<1e-3) 槽位2逐位不动=True | **fp16 原生 + 混宽**：`qdt=float16, adt=float32`（生产形状），计算量恒 fp32 | `ascend\|rope[ascend]\|h2d16s2g1qfloat16afloat32` |
| attn_sw | **PASS** rel=1.041e-07 (<1e-4) | **PASS** rel=5.555e-07 (<1e-4) | 入口升宽 | `ascend\|attn_sw[ascend]\|h2d32w8b16x16` |
| lora | **PASS** apply rel=0.000e+00，dx/da/db=0.0 | **PASS** apply rel=0.000e+00，dx/da/db=0.0 | 入口升宽（`lora_asc.py:48-50`）⇒ 内含 gemm 模具为 `dfloat32` | `ascend\|gemm[ascend]\|n16k32b16x16x32anonedfloat32` 等 4 副 |

**每格附加判别**（防"放宽阈值把静默错数据放过来"）：
- linear：三形状（m33 尾块 / m8+relu / 方阵 n=k=32）＋**错朝向**参考差 1.77e+00（>1e-2 才算"真的没翻"）；
  fp16 档另比"**每步降回 fp16 累加**"的错误口径＝2.28e-03 / 1.95e-01 / 1.10e-03 ≫ rel=0 ⇒ 归约确在 fp32。
- dW：**漏块**判别（只算前 16 个 token 的错误口径）=9.88e-01。
- readout：纯搬运 ⇒ 要求**逐位同**（`bitexact=True`），backward 与 `index_add_` 口径同（rel=0.0）。
- add_ln：保 `r_h<1e-5 / r_y<3e-4` 的**有偏/无偏判别**（无偏口径 rel=7.5~7.9e-03 明显分开）。
- rope：**槽位 2（值）逐位不动** ⇒ 就地语义未越界。
- attn_sw：参考侧未来权重严格为 0 ⇒ 窗外不参与；首位置输出非零 ⇒ 正文真跑。

## 3. 改动前后 fp32 全档对照（未回归凭据）

命令（改前/改后各跑一次，换 `TILELANG_CACHE_DIR`）：
```bash
cd release && TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --dtypes fp32
```
改后 9/9、改前 9/9，逐格 rel **完全同值**（linear 1.387e-07 / dw 2.258e-07 / readout 0 / add_ln 2.616e-07
/ gdn_fwd 1.298e-07 / gdn_bwd 1.955e-07 / rope 6.194e-08 / attn_sw 1.041e-07 / lora 0），唯一差异是**键尾
多了位宽段**（`...b16x64x32anone` → `...anonedfloat32`；`...g1` → `...g1qfloat32afloat32`）。仓内所有键
引用都是 `startswith`/子串匹配（如 `tests/test_ascend_gradcheck.py:58`），无全等断言 ⇒ 追加段安全。
fp16/bf16 六件未改的格（readout/add_ln/gdn×2/attn_sw/lora）改前后数字也逐字同 ⇒ 改动面精确落在三件。
本窗末尾又做了 2 处**纯空白**的签名续行对齐归一（`rope_asc.py:87` 21→18 空格、`gemm_asc.py:212`
19→18），归一后 fp32 9/9、fp16 9/9 与归一前**逐格同值**（`/tmp/c2_after2_fp32.txt` /
`/tmp/c2_after2_fp16.txt`），行号未漂移。

### 3.1 两条**独立工装**旁证（不用自己的探针给自己打分）

| 命令 | 真实输出 | 判什么 |
|---|---|---|
| `TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/verify_gemm.py --cpu` | `cpu act=relu allclose=True max_abs_diff=0.000e+00`、`cpu act=none ... max_abs_diff=0.000e+00` ⇒ `K-GEMM CPU-SEMANTIC-PASS` | **target=cpu 语义路一字未动**（`gemm_cpu_impl` 不参与位宽参数化） |
| `... verify_rope.py --golden` 与 `--golden --dtype float16` | 各 8 行 `I-ROPE-ASC-GOLDEN-PASS ... max_abs=2.38e-07 slot2_逐位不动=True`（含 sign=-1 反向） | 昇腾正文的**循环结构口径**在 I 代理的 numpy 影子件（尺子 `sys1.testing.torch_ref.rope_ref`）下未变 |

两条如实登记（都属计划外的连带面/界限）：

1. **键格式改动的附带面（计划外但必要）**：位宽段走的是 `ascend_env.make_key`，对 target=cpu 同样生效
   ⇒ cpu 侧键现在也带 `dfloat32`：实测
   `cpu_keys=['cpu|gemm[cpu]|n64k64b16x64x64areludfloat32', 'cpu|gemm[cpu]|n64k64b16x64x64anonedfloat32']`。
   键名变了、**cpu 正文与语义没变**（上面 `max_abs_diff=0` 直证）；仓内键引用全是 `startswith`/子串
   匹配（`tests/test_ascend_gradcheck.py:58`），无全等断言 ⇒ pytest 仍 340 passed。若后来工装做键全等
   断言，需知这一段是 P1-1m 起新增的。
2. **`verify_rope.py` 在 host 上必然 `I-ROPE-ASC-COMPILE-FAIL ... AssertionError: Target ascend is not
   supported`**（host tilelang 无 ascend 后端 ⇒ 这正是本窗必须搭 cpu-alias 轨的原因，不是回归）。且该
   工装 `--dtype` 会把 cos/sin **强制降成与 qkv 同宽**（`verify_rope.py:137-138`），所以它**测不到**
   "qkv fp16 + 表 fp32" 的生产混宽形状——混宽凭据只在 §2 rope 行（键 `qfloat16afloat32`）与 §7 卡面
   签名，不能拿本工装当混宽证据。

## 4. fp16 缺口的三处修法（只动 `*_asc.py` 的 ascend 正文）

| 件 | 改前为什么不行（一手） | 改了什么（行号） |
|---|---|---|
| `gemm_asc.py` | fp16 张量 + 正文声明 `"float32"` ⇒ **编译按声明位宽出核成功**，发射期 cython `_check_buffer_dtype` 抛 `ValueError: Buffer dtype mismatch for parameter A: expected torch.float32, got torch.float16`（抛点 `gemm_asc.py:292` `kernel(A, W, zero, C)`，在 `run()`@272 之内、**不在 `get_compiled` 的 try 内 ⇒ 生产直崩**） | 位宽参数化：`_ASC_DTYPES/_TL_DT` @63-65；`gemm_scalar_body(..., idt, odt)` @131-132 + `_ld/_st` @154-158；prim_func 声明改 `idt/odt` @160-162；`w_ub` DMA 装填缓冲改 `idt` @170；`gemm_asc_impl` 签名 @211-212、透传 @223-224；plan 白名单 @257；键位宽段 @268 |
| `gemm_bwd_dw_asc.py` | 同上（`parameter dY`） | `_ASC_DTYPES/_TL_DT` @46-48；`dw_scalar_body(..., idt)` + `_ld` @129；prim_func `dY/A` 改 `idt`、**`GW` 留 float32** @133-134；`dw_asc_impl` @180、透传 @191；plan 门 @222；键 @232；抛点 `gemm_bwd_dw_asc.py:251` `kernel(dY, A, GW)` |
| `rope_asc.py` | `_plan` 要求 `qkv.dtype == cos.dtype`，而生产的 cos/sin 出自 `train_step._build_params`（`torch.cos(angle)` ⇒ **fp32**）、主干 qkv 是 fp16 ⇒ **生产形状被 plan 判死、内核永不被调用**，forward 静默落 eager（改前 fp16 格实测 `NOKEY:rope[ascend]`） | `_ASC_DTYPES/_TL_DT` @38-40；`rope_asc_impl(..., qdt, adt)` + `_lq/_la/_sq` @117-123；prim_func 三张量各自位宽声明 @127-129；`_plan` 位宽门 @183-184（**补 `sin` 比对**、允许 qkv 与表**混宽**）；键 @192；`forward()`@214 里三条硬编码 fp32 门槛下放到 `_plan` 单点裁决 |

三条共同纪律：
1. **cast 只插在标量 load/store 面**，DMA 搬运不许顺带转类型（`tilelang/ascend/analysis/vf_checker.py`
   "DMA copies cannot perform type casting"）⇒ `w_ub` 与源同宽，`a_ub/dy_ub`（夹址标量填装面）留 fp32。
2. **算式与累加面恒 fp32**，出口不降位（dW 的 `GW`、gdn 五梯、lora 梯度都是 fp32 出口）⇒ 与
   `autograd_asc._work_dtype` 同口径。
3. **dtype 不符在 plan 期显式拒**（不再"编译门瞎、发射期崩/静默错数据"），且**缓存键带位宽段**
   （§6 证同形状换位宽必得新模具）。
4. `*_cpu_impl` 与 target=cpu 语义路**一字未动**；`idt/odt/qdt/adt` 缺省 `"float32"` ⇒ fp32 轨
   `_ld/_st/_lq/_la/_sq` 全是恒等函数、kwargs 与改前同值 ⇒ 生成码逐字不变（§3 数字 + §7 fp32 签名双向印证）。

补丁本体可复跑可回滚：`reconcile/C2_probe_apply_dtype.py`（30 条锚点定点，锚点数≠1 则整批不动）。

## 5. "fp16 PASS" 的确切含义（两种形态，不混谈）

| 形态 | 件 | 事实 |
|---|---|---|
| **fp16 原生内核** | linear / dW / rope | 内核按 fp16 声明**直接读 fp16 内存**，标量 load 处升 fp32 算、写回时降位；卡面签名 `__gm__ half*` 直证（§7） |
| **接口 fp16 / 内核 fp32（入口物化）** | readout / add_ln / gdn fwd / gdn bwd / attn_sw / lora | 入口先 `.float().contiguous()`（`lora_asc.py:48-50`、`add_ln_asc.py:407-408`、`gdn_asc.py:671-672`、`attn_sw_asc.py:297`、`letter_readout_asc.py:251`）再交正文 ⇒ 模具与位宽无关（§6 已证复用无害：两档数值均 PASS）。这**不是 fp16 内核**，也不该被写成"该件已支持 fp16 内核路" |

> 对本窗命题的意义：下窗拿 fp16 生产形状来测，**六件的 fp16 接口形状在 host 已真编真跑且对得上**，
> 三件（linear/dW/rope）的 fp16 **内核**路也已打通并编到卡上；fp16 不再是"只验过 fp32 的路径"。

## 6. 缓存键位宽隔离（同进程先 fp32 后 fp16，中间不 reset）

```bash
cd release && TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_key_isolation.py
```
`C2-ISO-SUMMARY 9/9 格双档闭环 失败=无`。要点：
- linear：fp32 三副 → fp16 **另新增三副**（`n32k32…dfloat16` / `n64k32…anonedfloat16` / `n64k32…areludfloat16`）；
- dW：新增 `…tc16b64x32dfloat16`；rope：新增 `…qfloat16afloat32`；
- 其余六件 fp16 档**新增=0**（复用 fp32 模具）且数值 PASS ⇒ 与 §5 的"入口物化"分型一致，复用是正确构造。

## 7. 容器 fp16 target=ascend 编译判决（`cann910b-i`，910B/dav-2201）

```bash
docker cp $WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h \
  cann910b-i:/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h
docker exec cann910b-i bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/C2_probe_card_fp16.py'
```
同步核对：trunk `port910b_compat.h` md5 `ca023c2e01746ca52c87f36bcea93ee9` 两侧一致（跑前容器内是旧版
`f56432b0…`，已按要求覆盖；旧件备份容器 `/tmp/port910b_compat.h.bak2`）。三件产品文件 host↔容器 md5 一致
（`/work` 即 host 的 release 目录，virtiofs 实时可见）：`gemm_asc.py=a0e7f19e5cd306c9a98750e804f549ec`、
`rope_asc.py=1d471af4f8f12327998b3c5719d016ab`、`gemm_bwd_dw_asc.py=0d4b448def98e8708a87c9008035453e`。
本窗在**最终文件形态**（含 §3 末尾 2 处纯空白对齐归一之后）**复跑过一次全量卡判决**，7 位判决行 + 7 份
签名与首跑逐字同，`C2-CARD-SUMMARY A=7/7 B(取证到手)=3/3`（stdout 另存 host `/tmp/c2_card_final.txt`）。

`C2-CARD-SUMMARY A=7/7 B(取证到手)=3/3 A_FAIL=无 B_UNEXPECTED=无`（判决行样例，签名行是位宽直证）：

| 位宽 | 判决 | 生成码 `__global__` 签名 |
|---|---|---|
| gemm fp32 | `C2-CARD-gemm[ascend]-fp32-COMPILE-PASS` keys=`[...anonedfloat32]` src=2854B | `(__gm__ float* A, __gm__ float* C, __gm__ float* W, __gm__ float* bias, int32_t m)` |
| gemm fp16 | `...-fp16-COMPILE-PASS` keys=`[...dfloat32, ...dfloat16]` | `(__gm__ half* A, __gm__ half* C, __gm__ half* W, __gm__ float* bias, int32_t m)` |
| dW fp32 / fp16 | 两位均 `COMPILE-PASS` keys=`[...dfloat32, ...dfloat16]` | fp16：`(__gm__ half* A, __gm__ float* GW, __gm__ half* dY, int32_t m)` ⇒ **出口 GW 恒 fp32** |
| rope fp32 | `COMPILE-PASS` keys=`[...qfloat32afloat32]` | `(__gm__ float* Cos, __gm__ float* QKV, __gm__ float* Sin, int32_t tokens)` |
| rope fp16 同宽 | `COMPILE-PASS` keys=`[...qfloat16afloat16]` | `(__gm__ half* Cos, __gm__ half* QKV, __gm__ half* Sin, ...)` |
| **rope fp16 + 表 fp32（生产混宽）** | `COMPILE-PASS` keys=`[...qfloat16afloat32]` | `(__gm__ float* Cos, __gm__ half* QKV, __gm__ float* Sin, ...)` |

全部 7 份生成码 `SimtVF=False`、`Cube/mad=False`（910B 标量面纪律未破），fp16 轨 `float_word` 仍 ≥3
⇒ **只有 GM 指针换了位宽，累加/UB 面仍是 fp32**（生产口径在生成码上的投影）。

## 8. bf16 缺件清单（**不得写成"支持"**）

### 8.1 一手卡面报错（本窗取证，绕开 plan 白名单直接按 bf16 组正文真编译）

```
RuntimeError: Ascend device compilation failed.
Command: .../ccec_compiler/bin/bisheng -O2 -fPIC -std=c++20 -DTL_PORT910B_NATIVE_TYPES ... \
         --npu-arch=dav-2201 -cce-aicore-only /tmp/<x>/tl_kernel.asc -o /tmp/<x>/tl_kernel.rel.o
Compiler output:
  fatal error: error in backend: not support bf16 type cast
  bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
  clang version 15.0.5   Target: aarch64-unknown-linux-gnu
Source:
  extern "C" __global__ __vector__ void gemm_impl_kernel(__gm__ bfloat16_t* A, __gm__ bfloat16_t* C,
                                                          __gm__ bfloat16_t* W, __gm__ float* bias, int32_t m)
```
三件同型：`C2-CARD-gemm-bf16-UNSUPPORTED` / `C2-CARD-gemm_dw-bf16-UNSUPPORTED` /
`C2-CARD-rope-bf16-UNSUPPORTED`，原文均为 `fatal error: error in backend: not support bf16 type cast`
（bisheng 15.0.5，dav-2201 **vector 面**，exit 70）。原始 stdout 在容器
`cann910b-i:/tmp/c2_bf16_raw.txt`（本窗一次取证运行），汇总在容器 `/tmp/c2_card3.txt`。
后端是 **backend abort**，bisheng 不给源码行列——能定位到的最小锚点是上面 `Source:` 里的
`__gm__ bfloat16_t*` 签名 + 正文里的标量 `Cast(→float)`（生成码里由 `_ld/_st/_lq/_la/_sq` 插出）。
既有旁证（评审引用面）：`ascend/port910b/attempts/B/b_addln_910b.py:24`（同一句原文，`dump_bf16.py` 取证）
与 `attempts/D/d_dw_910b.py:33`。

### 8.2 逐件现状（bf16 轨实测，host cpu-alias）

| 件 | bf16 判决（改后） | 一手表现 | 定位（行号） | 影响 | 替代方案 |
|---|---|---|---|---|---|
| linear | **UNSUPPORTED（显式拒）** | `plan` 返回 None ⇒ 落 `_eager`（无键） | `gemm_asc.py:63` 白名单 + `:257` 拒点；缺件依据注释 `:57-62`；cast 面 `:154-158` | 走不了内核，数值仍对（eager fp32 口径）⇒ **不崩但不上卡** | 生产走 fp16 |
| dW | **UNSUPPORTED（显式拒）** | 同上 | `gemm_bwd_dw_asc.py:46` + `:222`；注释 `:43-45`；cast 面 `:129` | 同上 | 生产走 fp16 |
| rope | **UNSUPPORTED（显式拒）** | `_plan` 返回 None（改前也是 None，但改前理由错成"表不同宽"） | `rope_asc.py:38` + `:183-184`；注释 `:33-37`；cast 面 `:117-123` | 同上 | 生产走 fp16 |
| readout | host PASS（rel=0 逐位同） | 未进 DSL：`letter_readout_asc.py:251` `.float()` 物化后正文恒 `"float32"`（`:49-50`/`:95-96`/`:131-132`） | 位宽真相=**内核无 bf16 路** | 声称"支持 bf16"是**假象**：bf16 只在入口被升宽 | 生产走 fp16；真要 bf16 内核需先补 bisheng 面 |
| add_ln | host PASS（r_y=0.00e+00） | 同上（`add_ln_asc.py:407-408`），正文声明 `"float32"`（`:114-138`） | 评审证据即本件的 bf16 尝试（`b_addln_910b.py:24`） | 同上 | 生产走 fp16 |
| gdn fwd / bwd | host PASS（0.00e+00 / 2.611e-07） | 同上（`gdn_asc.py:671-672`、`:708-709`） | — | 同上 | 生产走 fp16 |
| attn_sw | host PASS（0.00e+00） | 同上（`attn_sw_asc.py:297`） | — | 同上 | 生产走 fp16 |
| lora | host PASS（0.00e+00） | 同上（`lora_asc.py:48-50`），内含 gemm 模具 `dfloat32` | — | 同上 | 生产走 fp16 |

命令：`TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --dtypes bf16 --neg`
（`--neg` 同时打印位宽门 accept/kwargs/key：`C2-NEG-gemm.plan[bf16] accept=False`、
`C2-NEG-dw.plan[bf16] accept=False`、`C2-NEG-rope.plan[bf16/tab_fp32] accept=False`、
`C2-NEG-rope.plan[bf16/tab_bf16] accept=False`）。

### 8.3 为什么改前"看起来支持"是错的（两条，都已在产品件里堵掉）

1. **改前三件不是"崩得晚"就是"静默错数据"**：host 侧 `ValueError: Buffer dtype mismatch for
   parameter A/dY: expected torch.float32, got torch.bfloat16`（发射期 cython 校验）；卡侧编译门对
   位宽错配**是瞎的**（按声明位宽出核），runtime 一旦跳过校验（adapter 有 `skip_tensor_validation`
   形参）就成了 `reconcile/I_RESULT.md §⑥-3` 记的那类"编得过、出 .o、数据错"。现在 bf16 在 **plan 期**
   就被拒，且拒因写在件里（含取证出处），不会被后来者当"白名单漏加"重新放开。
2. **六件的 bf16 PASS 是入口物化**，与"该件支持 bf16 内核"无关；本表 §8.2 逐件标注，禁止合并计数。

## 9. 未闭环 / 已知缺口（如实）

1. **`rope_asc.backward()` 没有 ascend 分支** ⇒ target=ascend 的 rope 反向（含 fp32）落 eager。
   这是**既有路由缺口、非 dtype 面**，本窗未动件的路由结构，只在矩阵 rope 行如实标注
   （`反向 rel=0.00e+00 落 eager`）。修法应归 rope 路由那一格（另窗）。
   补充取证：`verify_rope.py --golden` 的 sign=-1 影子件 8 行全 PASS ⇒ 反向**算式**口径本身没问题，
   缺的只是 `_plan/_run` 那条路由，不是数值面。
2. **fp16 端到端未验**：`ascend/train_step.py:209-253` 当前全链 `out_dtype=torch.float32`、参数
   `torch.randn` 默认 fp32（B 代理写面，禁改）⇒ 本窗的 fp16 闭环是**件级/形状级**证据，不是
   train_step 端到端 fp16 证据。下窗若把主干切 fp16，件级路已就位（linear/dW/rope 原生、六件升宽）。
3. **fp16 真机数值未判**：容器无设备 ⇒ §7 只到编译；数值走 host cpu-alias（§2/§6）。
4. **bf16 内核面 = 缺件**（§8）：要么等 bisheng vector 面补 bf16 标量 cast，要么生产恒走 fp16；
   本窗选择**显式拒 + 清单**，未放开白名单。
5. **`C2_probe_apply_dtype.py --revert` 已不可用（如实）**：补丁落地后又做了 4 处**纯缩进美化**
   （函数签名续行 ×3 + gemm 模块 docstring 续行 ×1，均为空白改动、无语义；§3 末尾那 2 处对齐归一
   属同批行的再一次空白调整）。实测 30 条 EDITS 里有
   **3 条的 new 文本不再逐字同**（EDITS #8/#15/#24＝`gemm_asc_impl`/`dw_scalar_body`/`rope_asc_impl` 三处签名续行的缩进被后续美化过），
   `main()` 的 `if miss: return 2` 在**写文件之前** ⇒ `--revert` 报 `锚点异常 3 处（未做任何改动）`
   并整批拒绝（安全失败，不破坏文件；本窗**未实测执行** `--revert` 以免误改）。同理 `--check`
   现报 `锚点异常 27 处`——那是 **old 文本已不存在**＝补丁已应用的幂等保护，非缺陷。失配数可复跑：
   `.venv/bin/python -c "import sys;sys.path[:0]=['.','ascend/kernels/reconcile'];import C2_probe_apply_dtype as m;print([(i,f) for i,(f,o,n) in enumerate(m.EDITS) if m.read(f).count(n)!=1])"`
   → `[(8,'gemm_asc.py'),(15,'gemm_bwd_dw_asc.py'),(24,'rope_asc.py')]`（27/30 的 new 文本仍逐字在位）。
   **回滚请走 git**
   （本窗禁 git 写操作，未提交）。
6. **探针自身的两个假绿坑（已修，方法学记录）**：首版 `q(t,dt)=t.to(dt).float()` ⇒ 接口张量实为 fp32，
   linear/dw/rope 的"fp16"格跑的是 fp32 内核却判 PASS（`/tmp/c2_fp16_before.txt` 那份 8/9 因此**作废**，
   本窗改前数字一律以 `/tmp/c2_before.txt` 的 15/27 为准）；首版把参考量化到**接口 dtype** 而非**输出
   dtype** ⇒ 出口 fp32 的件（gdn 五梯）被引入纯量化噪声假红 `dq rel=2.631e-04`。修法见探针 docstring。

## 10. 复跑命令汇总

```bash
# host（.venv=release/.venv/bin/python；每条都换新缓存目录）
cd release
TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --dtypes fp32
TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --dtypes fp16
TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --dtypes bf16 --neg
TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py --rows rope --dtypes fp16   # 单格
TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_key_isolation.py
.venv/bin/python -m pytest tests -q -m "not integration"        # 340 passed 基线
.venv/bin/python ascend/kernels/reconcile/C2_probe_apply_dtype.py --check   # 锚点体检（§9.5：--revert 已失效，回滚走 git）
.venv/bin/python ascend/kernels/reconcile/verify_gemm.py --cpu                     # §3.1 cpu 语义路未回归
.venv/bin/python ascend/kernels/reconcile/verify_rope.py --golden                  # §3.1 rope 循环结构口径（host 编译行必 FAIL，只取 GOLDEN 行）
```
产物：本文件 + `C2_probe_dtype_matrix.py`（矩阵主探针）、`C2_probe_key_isolation.py`（键隔离）、
`C2_probe_card_fp16.py`（卡编译＋bf16 取证）、`C2_probe_apply_dtype.py`（定点补丁）、
`C2_probe_before.py` / `C2_probe_alias_shim.py`（改前取证与垫片可行性）；
被改产品件：`gemm_asc.py` / `gemm_bwd_dw_asc.py` / `rope_asc.py`（仅 dtype 所需最小改动）。
