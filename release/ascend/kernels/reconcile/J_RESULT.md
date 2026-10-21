# J_RESULT — P1-4 接口 home 910B 编译合流：旗舰件 GDN（执行代理 J）

**结论：三子项全部合流并出设备二进制。`ASC-COMPILE` 9/9 PASS + 静态口径 3/3 PASS（FAIL=0）；
`target=ascend` 经 bisheng 产出 3 份 `kernel.aibin` + 3 份 `executable.so`；`target=cpu` 语义回归
3/3 PASS（FWD/CONV 真走 cpu 模具）。反向 `BWD_STATUS` 由 `partial` 撤为 `kernelized`。零新 compat。**

---

## ① 完成情况（逐条附凭据）

复跑口径（容器 `cann910b-j`，release 挂 `/work`；每条前置）：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
export TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache.XXXXXX)   # 必须新目录，防陈旧 PASS
python3 /work/ascend/kernels/reconcile/verify_gdn.py           # 全家；--gate 只跑首推档
```

### 子项① delta-rule 前向 ascend 路径 SimtVF → 910B 可编形态 — DONE

| 例 | 判决原文（可复跑） |
|---|---|
| ub 档（首推） | `J-GDN-FWD  ub 档（首推） ASC-COMPILE-PASS key=ascend\|gdn[ascend]|h8k16v16|c8ubh .o~3699B [59行 expf=True simt=False mte=False gm_rmw=False null_ubuf=[]]` |
| pure 档（保留） | `J-GDN-FWD  pure 档 ASC-COMPILE-PASS key=ascend\|gdn[ascend]|h8k16v16|c8pureh .o~3342B [53行 expf=True simt=False mte=False gm_rmw=True]` |
| 标量门入参 (H,T) | `ASC-COMPILE-PASS key=ascend\|gdn[ascend]|h8k16v16|c8ubh`（与通道门同键 ⇒ 同一份产物） |
| 生产形状 H16 T128 DK64 DV64 cores16 | `ASC-COMPILE-PASS key=ascend\|gdn[ascend]|h16k64v64|c16ubh .o~3707B` |

载体：`T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial` + `T.alloc_var`，**零** `T.SimtVF`/`T.Parallel`/
`T.Pipelined`/`T.copy`（`grep -n "SimtVF\|T.Parallel\|T.Pipelined" gdn_asc.py` 只命中注释行 161-162）。
状态跨 token 串行、并行度只落在 (head, 值维列) 通道块上；exp 走 §11 软件 `expf`（落码 `expf=True`）。

### 子项② 反向去 partial（接入已证的 910B 标量六梯）— DONE

| 例 | 判决原文 |
|---|---|
| ub 档（首推，六梯） | `J-GDN-BWD  ub 档（首推，六梯） ASC-COMPILE-PASS key=ascend\|gdnb[ascend]|h8k16v16|c8ubz .o~7334B [101行 expf=True simt=False mte=False gm_rmw=False]` |
| pure 档（保留） | `J-GDN-BWD  pure 档 ASC-COMPILE-PASS key=ascend\|gdnb[ascend]|h8k16v16|c8purez .o~7592B [108行 expf=True gm_rmw=True]`（GM 就地 RMW ⇒ 真机待证 G-F3） |

* 六项梯度全实现：`dq / dk / dv / dg / dβ`（+ `dH0`，`return_dh0=True` 时交回）。
* `BWD_STATUS = "kernelized"`（`gdn_asc.py:43`），`gdn_kernel.py` 同步导出并在注释里写明档位与
  head 并行约束：验证 `J-STATIC-BWD-STATUS PASS gdn_kernel.BWD_STATUS='kernelized' gdn_asc='kernelized'`。
* 反向是**两趟流水**（前向重算落 Hist → 伴随递推），`run_backward` 自建并共用同一块 Hist。

### 子项③ 新增短卷积 conv 入口（R7 崩点）— DONE

| 例 | 判决原文 |
|---|---|
| scalar 档（首推，C=2048/L=512） | `J-GDN-CONV scalar 档（首推） ASC-COMPILE-PASS key=ascend\|gdnconv[ascend]|c2048k4b1|c8scalare .o~1689B [37行 expf=True simt=False mte=False gm_rmw=False]` |
| 批入参 (2,C,L) | `ASC-COMPILE-PASS key=ascend\|gdnconv[ascend]|c64k4b2|c8scalare .o~1865B` |
| ubstage 档（保留） | `ASC-COMPILE-PASS key=ascend\|gdnconv[ascend]|c64k4b1|c8ubstagee .o~2448B [41行 mte=True]`（走 MTE2/MTE3 ⇒ G-C4/G-C7 待证） |

`J-STATIC-CONV-ENTRY PASS gdn_kernel.conv_forward=<function conv_forward>` — 入口已在接口 home 挂上，
"conv 在上游之外"这条口径自 P1-4 起作废。

### 设备二进制（"出 .o" 的直接证据，非源码字节数）

```bash
python3 /work/ascend/kernels/reconcile/verify_gdn.py --gate   # 全新 TILELANG_CACHE_DIR
find $TILELANG_CACHE_DIR -name kernel.aibin -o -name executable.so
```

实测：`N_AIBIN=3 N_SO=3`；aibin 字节 `6400 6408 6392`，executable.so 字节 `49776 25192 37480`
（正好 fwd / bwd / conv 各一份；910B 工具链里设备码的落盘形态是 `ascend-binaries/<hash>/kernel.aibin`，
不是 `.o`）。

### target=cpu 语义不回归（宿主 `release/.venv`，cpu 后端可编）

```bash
cd $REPO_ROOT/release && .venv/bin/python -c \
 'import sys;sys.path.insert(0,".");import importlib.util as u;\
  sp=u.spec_from_file_location("v","ascend/kernels/reconcile/verify_gdn.py");\
  m=u.module_from_spec(sp);sp.loader.exec_module(m);\
  print(m.ascend_env.backend_available("cpu"));[print(*r) for r in m.cpu_checks()]'
```

```
cpu avail: True
J-GDN-FWD-CPU  PASS keys=True              # 真走 cpu 模具 cpu|gdn[cpu]|h2k8v8，vs 独立 bmm 尺子 ≤1e-4
J-GDN-BWD-CPU  PASS 五梯 vs autograd ≤1e-4 # cpu 路仍交 torch 自动微分（接口既有语义，未动）
J-GDN-CONV-CPU PASS keys=True              # 真走 cpu 模具 cpu|gdnconv[cpu]|c16k4b1，vs _eager 尺子
```

`gdn_cpu_impl` 正文一行未动（`T.copy` 仍只在 cpu 正文 121-124 行）。

---

## ② 错误与阻塞（含已解决的）

| # | 现象（原文关键行） | 位置 | 处置 |
|---|---|---|---|
| E1 | BWD 判 `PRECOMPILE-ERR（blockers=[]）` 但 feats 显示前向的 59 行产物 | `reconcile/verify_gdn.py` `_Dummy.__call__` | **已修**：判决件的 dummy 原本"调用即抛 `_LaunchSkip`"，而反向是两趟流水——第一趟（前向重算）一抛，第二趟伴随件永远进不了判决面。改成 dummy 只计数（`_STATS["launches"]`）不抛；判据仍靠 `compiled_keys()`，R14 不受影响。修后 BWD 出 101 行 / 7334B 真产物 |
| E2 | `ValueError: too many values to unpack (expected 2)` @ 批入参 conv 例 | `reconcile/verify_gdn.py drive_conv` | **已修**（驱动件自身 bug：传 `(2,64,128)` 却按两元解）：改 `x=f32(*shape); c=int(shape[-2])` |
| E3 | conv 的 cpu 路 `RuntimeError: shape '[16, 64]' is invalid for input of size 976` | `gdn_conv_asc.py _eager` | **已修**：`F.pad(左补 3)+valid conv` 已把长度带回 L，却又剪了 `[pad:pad+L]` ⇒ 少 k-1 格（976=16×61）。改回 `return F.silu(y)` |
| E4 | conv 的 cpu 模具开不出：`RuntimeError: Immutable variable 'acc' is used outside its defining region!` | `gdn_conv_asc.py conv_cpu_impl` | **已修**：cpu（eager）方言里"循环内累加变量"必须 `T.alloc_var` 在循环外声明——`gdn_cpu_impl` 正是此写法。补 4 个 `alloc_var`（顺手删掉未用的 `wt`）。修后 `cpu|gdnconv[cpu]|c16k4b1` 出现在 keys |
| B1 | 容器内 `方言后端 [cpu:probe] 不可用：TypeError: Forward references must evaluate to types. Got buffer.` | `ascend_env._compile_probe("cpu")` → `tilelang/language/eager/builder.py:1764` | **环境既有、与本波无关**（已证：`_compile_probe` 那份两行小件不含任何 GDN 代码，同样抛此错）。⇒ target=cpu 的语义回归改到宿主 `.venv` 跑（见上）；本件禁改 `ascend_env.py` |
| B2 | `ASC-gdn PRECOMPILE-ERR 未达 tilelang.compile（形状/plan 拒）`（统一 harness `compile_all_asc.py`） | `ascend/compile_all_asc.py:39-57` | **harness 缺口、非 plan 问题**（已证，见下"上报"段 G2）：同一 harness 跑 rope 也是 `PRECOMPILE-ERR`；把我的件按 harness 入参手工驱动，`plan` 正常出键 |
| B3 | `assert 'kernelized' == 'partial'` → `tests/test_ascend_gradcheck.py::test_gdn_cpu_backward_declared_partial_but_grads_are_correct` FAILED | `release/tests/test_ascend_gradcheck.py:303` | **待裁决（测试不在白名单，未改）**。失败只挂在口径断言的第一行，函数体后半段（梯度 vs autograd）没机会跑；等价证据已由 `J-GDN-BWD-CPU PASS 五梯 vs autograd` 提供。见上报 G1 |
| — | `Warning: Python Z3 scheduler failed ... falling back to topological sort` / `TVMScript printer falls back to the basic address printer` | tilelang 编译日志 | 无害告警：F/C 参照件基线复跑同样出现，判决不受影响 |

---

## ③ 疑惑点与自行决策（未问而自定的事，全部申报）

1. **decay 门 `g` 的语义分歧 → 推广为 per-channel（最重要的一条）**
   参照件 `attempts/F` 的 `g` 是**每 token 一个标量**，而接口 home 的 `gdn_cpu_impl`/`_eager`/
   `tests/test_ascend_gradcheck.py` 用的都是 **per-channel `(heads,seq,dk)`**。任务书明令"保留 `*_cpu_impl`
   与 target=cpu 语义对拍不动"⇒ 不能把接口改成标量门。决定：**把 F 的 DSL 逐位置推广到每键通道**
   （前向 `dec_ub[i]=T.exp(G[h,t,i])`；反向 `dg_t[i]=a_i·Σ_j S_{t-1}[j,i]·dS'_t[j,i]`），
   同时让入口 `_gate3()` 接受两种形状（`(…,H,T,DK)` 原样 / `(…,H,T)` 沿通道摊开），
   所以 harness 的 `_f32(2,8)` 标量门也能进模具。**代价（性能注记，非编译问题）**：
   `expf` 发射量从 `H·T·DV` 升到 `H·T·DV·DK`，卡上真跑时要按 `H%cores` 与这条重估。
2. **conv 入口的接口决策**：新增姊妹件 `gdn_conv_asc.py`，由 `gdn_kernel` 统一导出
   `conv_forward / conv_plan / conv_run / CONV_KERNEL / CONV_ASC_VARIANTS`。
   * **不**并进 `gdn_asc.py`：递推件吃的是"已卷过、已归一"的 q/k/v，形状（(C,L) 通道优先 vs
     (heads,seq,dk) 时间优先）与生命周期都不同，混在一起 `plan` 的判据会变成两套语义的并集、出事无法归因。
   * 形状：吃 `(..., C, L)`，`(C,L)` 与 `(B,C,L)` 都收（前导轴并进批轴）；`w` 收 `(C,4)` 或 Conv1d 的 `(C,1,4)`；
     `bias` 可缺省（件内补零，模具的张量清单里 bias 恒在场）。
   * 硬判据：窗宽必须 =4、三维 `w` 的中间维必须 =1（否则不是 depthwise）、通道数与 `x` 对不上、
     `y` 与 `x` 不同形、非 fp32 或不连续 ⇒ 一概不开模，落 `_eager`。
   * 档位：`scalar` 首推（零 UB/零 MTE，`attempts/C` 真机数值 6.35e-08 就是这一档），`ubstage` 保留。
   * `silu` 走 `z/(1+exp(-z))`（⇒ expf）；另留 `rational` 兜底档，**只对保编译不对数值**，docstring 已标。
3. **`BWD_STATUS` 取值**：新串 `"kernelized"`（原口径只有 `"partial"` 一个值可用，撤 partial 后需要正面
   描述状态），并把 `BWD_GRADES=("dq","dk","dv","dg","dbeta")` 与 `backward` 返回顺序对齐；
   `dH0` 只在 `return_dh0=True` 时追加在末尾。
4. **内部工作缓冲不外溢**：`H0/Hist/Sout/Lam/DH0` 全部件内现领，`forward/backward/plan/run` 对外的
   张量清单保持六路不变。`run_backward` 自建并共用 Hist（两趟必须同一块照片），docstring 明写
   "不能借道 `run()`"。
5. **`dstate` 缺省 `"zero"`**（截断 BPTT 口径）；`gm` 档在件里保留但未接进接口（与 pure 档同属待卡面）。
6. **核数均分**：前向 `units=heads*dv`、反向 `units=heads`（G-F2：反向必须按 head 切）、conv `units=chans`，
   三处各自 `_pick_cores`，挑不出整除退到 1 核；上限 env `SYS1_GDN_BLOCKS`（缺省 8）。
7. **动态长度**：沿用 `rope_asc` 合流先例用 `T.dynamic("seq"/"length")`，缓存键不含长度轴。
8. **`verify_gdn.py` 落在 `reconcile/` 下**（与同波 `verify_rope.py` 等 5 件一致），不是 `kernels/`；
   同步改了件内的 `RELEASE_ROOT` 推断与 docstring 用法路径。
9. **`tasks.md` 未回勾**：P1-4 条目在本波**写入白名单之外**，且同目录正被多个并发代理同时改写
   （`reconcile/` 已有 H_/I_ 战报），为避免互踩把勾选权交回编排者——请由主流程回写。

---

## ④ 偏离记录（与计划/任务书不一致之处）

| 偏离点 | 计划原文 | 实际做法 | 理由 |
|---|---|---|---|
| 判决件的 dummy 行为 | "monkeypatch `tilelang.compile`（真编后 dummy 抛 Skip）" | dummy **不抛**，只记 `launches` 计数 | 见 E1：两趟流水下抛在第一趟，伴随件根本进不了判决面（我一度拿到"PRECOMPILE-ERR 但 blockers=[]"的假阻塞）。判据本体（`compiled_keys()` 含 `ascend\|`）一字未松 |
| 前向 decay 面 | 照 F 件"每 token 标量门"搬 | 推广成每键通道 exp（`expf` 量级 ×DK） | 见决策 1：接口/cpu/测试都是 per-channel，改接口=动 cpu 语义=违令 |
| pure / ubstage 档 | "首推 ub 档；pure 的 GM RMW 跨 token 可见性 CPU 证不了，真机列待证" | 纳入**编译**面并各自 PASS（`gm_rmw=True`/`mte=True` 特征如实打出），但不入门禁（`--gate` 只含 ub/scalar/标量门三例） | 任务书允许"编译合流可含"；门禁 exit code 只认首推档，避免把待卡面当已证 |
| conv 数值尺子 | C 件真机 rel=6.35e-08 | 本波只出"编译 PASS + cpu 模具 vs `_eager` 一致"；ascend 侧数值 rel 待卡 | 波次边界（数值真机 rel 待卡） |
| `compat_patch_J.h` | "缺口写 `reconcile/compat_patch_J.h` 上报" | **未生成该文件** | 本波零新 compat：exp 走 trunk 已并的 §11 软件 `expf`（口径(a)），落码 `expf=True` 即证。G1/G2 两个缺口都不是 compat 层（分别是测试口径与 harness stub），写进本文件即可 |

---

## ⑤ 上报缺口（本波不动别人地盘，请编排者裁决）

* **G1（测试口径，需回写）**：`release/tests/test_ascend_gradcheck.py:303`
  `assert gdn_kernel.BWD_STATUS == "partial"` —— 撤 partial 后必挂（实测 `1 failed, 1 passed`）。
  建议改法：断言翻成 `BWD_STATUS == "kernelized"`（或参数化 `{partial, kernelized}`），并把该用例
  后半段"梯度 vs autograd"独立出来（现在被第一行断言挡住没跑）；等价证据在 `J-GDN-BWD-CPU`。
* **G2（统一 harness，禁改件）**：`ascend/compile_all_asc.py::_install_stubs` 只 stub 了
  `backend_available`/`_npu_present`，**没 stub `active_backend` 的设备门**（ascend target + CPU 张量 ⇒
  判 `torch_eager`），于是所有"经入口驱动"的件都判 `PRECOMPILE-ERR`。凭据：
  `python3 /work/ascend/compile_all_asc.py gdn rope` → `ASC-gdn PRECOMPILE-ERR / ASC-rope PRECOMPILE-ERR /
  汇总 PASS=0 FAIL=2`；而按 harness 同款入参手工驱动我的件：
  `GATE active_backend: torch_eager | plan-> gdn[ascend]|h2k16v8|c8ubh` —— **plan 正常出键**，
  且 `_gate3` 已兼容 harness 的 `g=_f32(2,8)` 标量门。⇒ 缺口在 harness 侧（补一行
  `ascend_env.active_backend = lambda *a, **k: TILELANG` 即可），我的件不需要为此改。
  另注：harness 的 `_Dummy` 同样"调用即抛"，因此它即使过了设备门，也只能判到**单趟件**
  （对 GDN 反向这种两趟件会重现 E1）。
* **G3（观察，非本波地盘，只读未动）**：`rope_asc.py:196` 是 `if name == ascend_env.TARGET_CPU:` 并且
  `_plan/_run` 都在这个分支里 ⇒ **经入口驱动时 ascend 路根本不会编译**。这也说明 harness 里
  `ASC-rope PRECOMPILE-ERR` 有**两重**独立原因（G2 的设备门 + 本条的入口分支），
  因此 G2 不能只靠 rope 这一例归因给 harness——我自己的 `ASC-gdn` 那一例是单因（已用手工
  `plan` 出键排除）。若编排者要统一修 harness，请连带评审入口件的 ascend 分支写法是否齐。

---

## 真实调用签名（`print_signatures()` 原文，供上层接线）

```
gdn_asc.forward(q, k, v, g, beta, out_dtype: torch.dtype = torch.float16, target: str | None = None, **spec_opts) -> torch.Tensor
gdn_asc.backward(q, k, v, g, beta, dout, target: str | None = None, return_dh0: bool = False, **spec_opts) -> tuple[torch.Tensor, ...]
gdn_asc.plan(q, k, v, g, beta, o, target=None, variant=None, blocks=None, emit_hist=None) -> dict | None
gdn_asc.run(q, k, v, g, beta, o, spec) -> bool
gdn_asc.plan_backward(q, k, v, g, beta, o, target=None, variant=None, blocks=None, dstate="zero") -> dict | None
gdn_asc.run_backward(q, k, v, g, beta, dout, dq, dk, dv, dg, dbeta, dh0, spec) -> bool
gdn_conv_asc.conv_forward(x, w, bias=None, out_dtype=None, target=None, **spec_opts) -> torch.Tensor
gdn_conv_asc.plan(x, w, bias, y, target=None, variant=None, blocks=None, silu=None) -> dict | None
gdn_conv_asc.run(x, w, bias, y, spec) -> bool
```

`spec_opts` 透传：递推件 `variant/blocks/emit_hist`（+ `backward` 侧 `dstate`），卷积件 `variant/blocks/silu`。
env 开关：`SYS1_GDN_VARIANT` / `SYS1_GDN_BLOCKS` / `SYS1_GDN_EMIT_HIST` / `SYS1_GDN_CONV_VARIANT` / `SYS1_GDN_CONV_SILU`。

---

## 待卡清单（本波边界外，诚实移交）

1. 三件的**真机数值 rel**（fwd/bwd 对照 `_eager`+autograd 双锚；conv 对照 C 件 6.35e-08 口径）。
2. `pure` 档：GM 就地 RMW 的跨 token 读后写可见性（G-F3），落码特征 `gm_rmw=True`。
3. conv `ubstage` 档：MTE2/MTE3 的单位与 stride 是否真被正确下发（G-C4/G-C7），特征 `mte=True`。
4. per-channel `expf` 的实际代价（`H·T·DV·DK` 次），以及 `H%cores` 在小头数（如 heads=2）下的带宽利用率。
5. HIST 缓冲带宽（G-F4 ≈33.5MB @ H16T128）与 `emit_hist=False` 纯前向推理档的收益。

---

## 写后三连（最终态）

```
wc -l  gdn_asc.py 776 | gdn_conv_asc.py 321 | gdn_kernel.py 43 | reconcile/verify_gdn.py 328
py_compile  ALL-PYCOMPILE-OK（四件）
grep  SimtVF/T.Parallel/T.Pipelined/annotate_buffer_versions → 仅注释行，代码零残留
      T.copy → 仅 gdn_cpu_impl(121-124，未动) 与 conv ubstage 保留档(168,175)；两个 ascend 递推正文零 T.copy
      BWD_STATUS = "kernelized"（gdn_asc.py:43）/ gdn_kernel.py:20 re-export
```
