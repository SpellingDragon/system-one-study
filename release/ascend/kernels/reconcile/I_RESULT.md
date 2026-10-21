# I_RESULT — P1-4 接口 home 910B 编译合流（代理 I：rope_asc / attn_sw_asc）

判决日期：2026-10-10 · 容器 `cann910b-i`（release 挂 `/work`）· patched 910B tilelang +
`ASCEND_NPU_ARCH=dav-2201` + `BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler`。
就绪取证：`/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`
= **971 行**（`ENV-COMPAT-IN-SYNC`）、`patch_bisheng.py` → `bisheng.py already patched`、
基线参照件复跑 `E-ROPE-COMPILE-PASS`。**本波只判编译，数值 rel 待卡窗**（任务书波次边界）。

## ① 结论（两件 target=ascend 均出 .o）

| 件 | 判决 | 凭据（复跑命令见 §⑤） |
|---|---|---|
| `rope_asc` | **PASS 16/16**（编译 8/8 + numpy 影子件 8/8），exit=0 | `—— rope 判决 PASS=16/16 ——`；4 形状串 × sign{+1,-1} |
| `attn_sw_asc` | **PASS 10/10**（编译 5/5 + 影子件 5/5），exit=0 | `—— attn_sw 判决 PASS=10/10 ——`；5 形状串 |
| rope fp16 附加档 | PASS 2/2（见 §⑥-3 的隐忧，不是好消息） | `--tokens 32 --dtype float16` → `—— rope 判决 PASS=2/2 ——` |
| CPU 语义面 | **不回归** | `.venv/bin/python -m pytest tests/test_ascend_gradcheck.py -k "rope or attn_sw" -q` → `5 passed, 1 skipped, 15 deselected`（与改前基线逐字一致） |
| compat 实付 | **零新增**：不需要 `compat_patch_I.h`（本目录未创建该文件） | rope 发射体 `math=[]`；attn 仅 `math=['expf']`（compat §11 既有软件件，P1 真机 max_rel_err=1.103e-06） |

### 编译判决原文（逐件）

rope（4 形状 × 2 符号；列默认档与尾行档）：
```
I-ROPE-ASC-COMPILE-PASS shape=t32h8d64s2 sign=+1 key=ascend|rope[ascend]|h8d64s2g1 .o_src~2137B
  asc_*=['asc_get_sub_block_id','asc_init'] has_ASC_IS_=False has_Simt_thread_carrier=False
  has_asc_sync=False has_gm_bypass=True has_mix=False has_mte_copy=False has_ubuf=False lines=30
  math=[] tl::*=['tl::read_gm_bypass_dcache','tl::write_gm_bypass_dcache']
I-ROPE-ASC-COMPILE-PASS shape=t32h8d64s2 sign=-1 key=ascend|rope[ascend]|h8d64s2g-1 .o_src~2191B ... lines=31
I-ROPE-ASC-COMPILE-PASS shape=t33h8d64s2 sign=±1 key=ascend|rope[ascend]|h8d64s2g±1        ← tokens 不被 8 整除，同 key 复用
I-ROPE-ASC-COMPILE-PASS shape=t8h4d32s2 sign=±1 .o_src~2129/2183B ；shape=t5h2d16s1 sign=±1 .o_src~1914/1966B（slots=1 档）
```

attn_sw（5 形状串）：
```
I-ATTNSW-ASC-COMPILE-PASS case=h4s64d32w8  key=ascend|attn_sw[ascend]|h4d32w8b16x16 .o_src~2925B
  has_Simt_thread_carrier=False has_mte_copy=False has_ubuf=True has_gm_bypass=True
  lines=46 math=['expf'] tl::*=['tl::write_gm_bypass_dcache'] ub_named_null_handles=[]
I-ATTNSW-ASC-COMPILE-PASS case=h4s33d32w8  key=ascend|attn_sw[ascend]|h4d32w8b16x16 .o_src~2925B  ← 与 s64 同 key：换长度不换产物
I-ATTNSW-ASC-COMPILE-PASS case=h2s70d16w16 .o_src~2920B；case=h1s64d64w64(window==seq) .o_src~2500B；
  case=h2s24d16w64(window>seq，夹位路) .o_src~2920B   —— 五行同格式，全 PASS
```

影子件原文（尺子取 P1 冻结件 `sys1.testing.torch_ref.rope_ref` / `attn_sw_ref`，**不用**本模块自带 `_eager`）：
```
I-ROPE-ASC-GOLDEN-PASS   tokens=32 heads=8 dim=64 slots=2 sign=-1 max_abs=2.38e-07 slot2_逐位不动=True  （8/8 同格式，±1 一致）
I-ATTNSW-ASC-GOLDEN-PASS case=h4s64d32w8  max_abs=5.96e-07 窗外扰动逐位不动=True 该变的真变了=True
I-ATTNSW-ASC-GOLDEN-PASS case=h4s33d32w8 max_abs=4.17e-07 · h2s70d16w16 3.28e-07 · h1s64d64w64 5.96e-07 · h2s24d16w64 3.58e-07
```
影子件划界：校的是**跨步领行不重不漏 / 就地双写不互污 / 槽位 2 不动 / 窗外严格 0 / 下标夹位不改数值**这几件
结构事实（`max_abs~1e-7` 量级来自影子件用 fp64 累加、尺子用 fp32）；**不**证明 DSL+compat `expf` 的真机数值——
那需上卡。attn 合流的是 attempts/E 的 **stage** 形态，其真机 rel=1.41e-07 已在 P1-1c 收过，故上卡只需复验
本波新增的两处（动态 `seq` 跨步领行 + 下标夹位）。

## ② 910B 方言映射（本次重写的实质）

| 被删的 950 载体 | 910B 替代形态 | 发射体证据 |
|---|---|---|
| `T.SimtVF(...)` + 其内 `T.Parallel`（910B 整族不存在，实测 COMPILE-FAIL） | `T.Kernel(n)` + `with T.Vector():` + 纯 `T.serial` | 两份 codegen 均 `has_Simt_thread_carrier=False`（`SimtVF`/`asc_vf_call`/`threadIdx` 三特征 0 命中） |
| 线程号取行（`vec_id` 逐元素） | **每核跨步领行**：`for r in T.serial(T.ceildiv(tokens-bx, NUM_BLOCKS))`、`t = bx + r*NUM_BLOCKS` | `for (r=0; r<(((tokens+7)-block_idx)>>3); ++r)`；免越界守卫（数学上恰好覆盖 `[0,tokens)`：被除数恒非负、`t_max ≤ tokens-1`） |
| `sinf/cosf` 硬件缺件（G-E1） | **表加载定案**：cos/sin 由 host 预计算、从 GM 载入；rope 本体只余乘加 | rope `math=[]`、`has_ubuf=False`（纯 GM 标量读写，连 UB 都不需要） |
| `exp` 硬件缺件 | trunk compat §11 软件 `expf`（DSL 侧 `T.exp`） | attn `math=['expf']` |
| 并行归约 / reducer / `T.Pipelined` | 标量面四趟串行：打分入 UB → 行最大 → `exp`+分母 → 加权除 `max(l,FLOOR)` | attn 46 行；`buf_dyn_shmem` 两段 `[0,win)` 与 `[win,2win)` 不重叠；`ub_named_null_handles=[]`（避开 G-C8"恰好 1 个被访问 shared ⇒ 命名空句柄"反例） |
| 向量化 store | 纯标量 `tl::write_gm_bypass_dcache` | 两件 `has_gm_bypass=True`、`has_mte_copy=False`（无 MTE copy，与 attempts/E 同形态） |

attn 相对 attempts/E 的本波新增两处：**动态 `seq`**（`T.dynamic("seq")`，故换长不换 key，见 h4s64/h4s33 同 key）
与**窗内下标夹位** `T.min(lo+j, seq-1)`（`window > seq` 时窗外格不落空指针），发射体见
`min((max(rr*8+(block_idx&7) - 7), 0) + j, seq-1)`。头基址用动态 seq 走跨步：`((block_idx>>3) * seq) * dim`。
可见性条件 `j < n` 被 codegen 折叠为 `j <= i`：在 `j ∈ [0,win)` 定义域内与 `j <= min(i, win-1)` 等价（同 G-E4 那类折叠）。
rope 反向（`sign=-1`）不发 `sinf/cosf`，只在载入后 `sn = 0.0 - sn`，发射体见 `sn = (float(0.0) - sn)`。

## ③ 真实接口签名与张量 I/O（未改动，逐字核对）

```python
# rope_asc.py   （ROTATE_SLOTS = (0, 1)；就地写回并返回同一对象）
def forward(qkv, cos, sin, rotate_slots: tuple[int, ...] = ROTATE_SLOTS, target=None) -> qkv(就地)
def backward(gqkv, cos, sin, rotate_slots=ROTATE_SLOTS, target=None) -> gqkv(就地)   # 与 forward 同路，sign=-1
def _plan(qkv, cos, sin, slots:int, sign:int, target) -> spec|None
    # 门槛：qkv 为 4 维且 size(1)==3；三路连续；dim 偶；slots ∈ (1,2)；
    #       cos/sin 形状均为 (tokens, dim//2)；
    #       target=cpu → qkv/cos/sin 全需 fp32；target=ascend → qkv.dtype ∈ {fp32,bf16,fp16} 且 qkv.dtype==cos.dtype
    # key = f"rope[{name}]|h{heads}d{dim}s{slots}g{sign}"；impl = rope_asc_impl（ascend）/ rope_cpu_impl（cpu）
def _run(qkv, cos, sin, spec) -> bool
# 张量 I/O：qkv (tokens,3,heads,dim)；cos/sin (tokens, dim//2)；dim 奇数 → forward 抛 ValueError

# attn_sw_asc.py
def forward(q, k, v, window:int, scale=None, out_dtype=torch.float16, target=None) -> lead+(seq,dim)
    # q/k/v (...,heads,seq,dim)；scale 默认 dim**-0.5；内部 `.float()` 物化 → plan/run → .to(out_dtype)
def forward_weights(q, k, window, scale=None, target=None)      # 未动
def plan(q, k, v, o, window:int, scale:float, target) -> spec|None
    # 四条硬门槛：① 三路 (heads,seq,dim) **fp32** 同形且与 o 同形、四路连续；② dim % 16 == 0；
    #             ③ window >= 1；④ scale 为有限正实数。任一不过回 None（seq 不要求被块宽整除）
    # key = f"attn_sw[{name}]|h{heads}d{dim}w{window}b{bm}x{bn}"
def run(q, k, v, o, spec) -> bool
def attn_sw_cpu_impl(...) / _eager(...) / _mask(...)             # 未动；target=cpu 语义对拍路径未动
def rope_asc_impl(QKV, Cos, Sin, heads:int, dim:int, slots:int, sign:int)          # ← 本次重写
def attn_sw_asc_impl(Q, K, V, O, heads, dim, window, scale, bm, bn)                # ← 本次重写
    # Q/K/V/O 均 (heads, seq, dim) fp32；整窗因果口径（任意掩码仍走 eager），未偷换
```
保留：`NUM_BLOCKS=8`、`NEG=-1e30`、`FLOOR=1e-30`、`BLOCK_Q/BLOCK_KV`（被 `attn_sw_kernel` re-export）。
删除：两件自有的 `VEC_THREADS/NUM_STAGES`（SIMT+流水线载体的残留常量）——全树**跨模块引用 0 命中**
（`grep -rn "rope_asc\.\(VEC_THREADS\|NUM_STAGES\)\|attn_sw_asc\.\(VEC_THREADS\|NUM_STAGES\)"` → 0；
`gemm_asc.py`/`gemm_bwd_dw_asc.py` 里的同名常量是它们自己定义的，与本件无关）。
`bm/bn` 作形参保留（进 key `...b16x16`）但 910B 标量形态不再分块消费——见 §⑥-4。

## ④ 写后三连

```
wc -l  rope_asc.py=243  attn_sw_asc.py=347
python3 -m py_compile ascend/kernels/{rope_asc,attn_sw_asc}.py ascend/kernels/reconcile/verify_{rope,attn_sw}.py → PYCOMPILE-OK
grep -n "SimtVF\|T\.Parallel" rope_asc.py attn_sw_asc.py → 0 命中（grep_exit=1）
```
（含 docstring：为免污染验收 grep，docstring 改用"950 代际 SIMT 向量载体"等描述性说法。）

## ⑤ 复跑命令（逐条判决都可复现）

```bash
# 0) 容器就绪取证（compat 行数应为 971）
docker exec cann910b-i bash -c 'wc -l /usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h'

# 1) rope：编译判决 + 影子件（期望 —— rope 判决 PASS=16/16 ——）
docker exec cann910b-i bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 \
    python3 ascend/kernels/reconcile/verify_rope.py --golden'
#    fp16 附加档（§⑥-3 证据）：… verify_rope.py --tokens 32 --dtype float16

# 2) attn_sw：编译判决 + 影子件（期望 —— attn_sw 判决 PASS=10/10 ——）
docker exec cann910b-i bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 600 \
    python3 ascend/kernels/reconcile/verify_attn_sw.py --golden'

# 3) 发射体留档：加前缀 TL_I_DUMP=/tmp/i_rope_kernel.cxx（或 i_attn_kernel.cxx）即 dump codegen
# 4) CPU 语义回归（期望 5 passed, 1 skipped）
cd release && .venv/bin/python -m pytest tests/test_ascend_gradcheck.py -k "rope or attn_sw" -q
# 5) 总判决 harness 交叉验证（当前期望两件 PRECOMPILE-ERR，归因见 §⑥-2）
docker exec cann910b-i bash -c '… cd /work && TILELANG_CACHE_DIR=$(mktemp -d) \
  python3 ascend/compile_all_asc.py rope attn_sw'
```
缓存纪律：每次判决换 `TILELANG_CACHE_DIR=$(mktemp -d)`（tilelang 按 kernel 源码哈希命中、不含模板内容，
不换就是陈旧 PASS——LOCAL_RUN.md/B/C/E 实证）。两份 verify 均**不得**加 `from __future__ import annotations`
（eager builder 的 `get_type_hints` 会把闭包变量当模块全局名 → NameError，C 波实测）。

## ⑥ 缺口上报（均**不在**本代理写入白名单，逐条附归因与运行时证据）

1. **`rope_asc.forward/backward` 没有 ascend 分支**（在本文件内，但改路由违反任务书"**只重写 `*_ascend_impl`**"）：
   `if name == ascend_env.TARGET_CPU:` 之后直落 `_eager`。运行时取证——把 `backend_available / _npu_present /
   active_backend` 三个门**全部**绕开后仍：`rope_asc.forward(qkv, cos, sin, target="ascend")` →
   `compiled_keys=[] blockers=[]`，即一次 `tilelang.compile` 都没发。故 `target="ascend"` 目前语义上等价于 eager。
   **待裁决**：是否由后续波次给 forward 补 ascend 分支（最小时延=照抄 CPU 分支的 fp32 物化 + 写回口径）。
   本波改用 `_plan/_run`（接口自身的 ascend 入口）判编译，等价入口、不改路由。
2. **总判决 harness 交叉验证：`ASC-rope PRECOMPILE-ERR / ASC-attn_sw PRECOMPILE-ERR / —— 汇总 PASS=0 FAIL=2 ——`**
   ——**不是**方言编不出，而是 harness 自身两处（`compile_all_asc.py` 禁改，报 harness owner）：
   - `_install_stubs()` 只打了 `backend_available/_npu_present`，**没打 `active_backend`**；而
     `active_backend("ascend", cpu_tensor.device)` → `torch_eager`（源码明文"昇腾产物只能在 npu 设备上发射"；
     实测：带 `device=cpu` 判 `torch_eager`，`device=None` 才判 `tilelang`）。harness 的张量全在 CPU，
     于是 `forward(target="ascend")` 永不调 `plan/run` → PRECOMPILE-ERR。**补一行 stub 即可**（本波 verify 已如此打桩并 16/16+10/10 全绿）。
   - `_drive_rope` 用 **fp16 qkv + fp32 cos/sin**，被 `_plan` 的"`qkv.dtype == cos.dtype`"门槛拒
     （实测 `_plan=None`；同 dtype fp32 则出 spec）。属 harness 驱动形状与接口判据不一致。
3. **编译门对"位宽错配"是瞎的（新发现的真实风险，需卡窗或接口判据定夺）**：`_plan` 在 ascend 侧允许
   fp16/bf16，而 `rope_asc_impl` 的 prim_func 声明的是 `"float32"`。实测 `verify_rope.py --tokens 32
   --dtype float16` 照样 **PASS 出 .o**（`I-ROPE-ASC-COMPILE-PASS ... .o_src~2137B`）——编译按声明位宽出核，
   fp16 张量到发射期才会被当 fp32 解读（**静默错数据**，不是编译错）。P1-1b 既有遗留；两条出路：接口 `_plan`
   收紧成"ascend 也要 fp32"，或 prim_func 按 `qkv.dtype` 参数化声明。本波未动（超出"只重写 impl"白名单）。
   对照：`attn_sw_asc.plan` 已强制 fp32（门槛①），`forward` 用 `.float()` 物化，无此隐患。
4. **attn 的 UB 宽度 = window**：`sc_ub/ps_ub` 共 `2*window*4B`（window=64 → 512B，本波 `h1s64d64w64` /
   `h2s24d16w64` 两档已编过）；更大 window（128/256）对 UB 容量的压力**未经真机核**，备选形态=按 `bn`
   分块 + online softmax 修正。`bm/bn` 现在是"进 key 不进正文"的形参。
5. **`ascend_env._compile_probe` 自身仍是 950 SIMT 形态**（白名单外、禁改）：真机上该自检探针可能判否，
   从而在**有卡**环境把两件都拖回 eager。建议列入 P1-4 后续波或 env owner。

（P1-4 的九件总目标未由本代理完成，`openspec/.../p2-13-ascend-runtime/tasks.md` 的 P1-4 行**未勾选**——
白名单外，且本代理只占其中 rope / attn_sw 两件。已交付的 `ASC-*` 两件转绿依赖 §⑥-2 的 harness 补桩。）
