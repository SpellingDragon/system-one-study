# compat_gap_E.md — 910B（dav-2201）port910b_compat 缺口登记（rope / attn_sw 波）

执行代理 E / p2-13-ascend-runtime 甲路 P1-1。容器 `cann910b-e`，CANN 8.5.0 + bisheng(clang) 15.0.5，
pip tilelang 0.1.15，注入态 `-DTL_PORT910B_NATIVE_TYPES -I<asc/impl> -I<asc/include>`
（`patch_bisheng.py` 本波首次注入该容器，凭据 `C-BISHENG-PATCHED`），
`ASCEND_NPU_ARCH=dav-2201 target_format=aibin`。

**compat 消费态**：v3 真源 **944 行**（B/C GAP 块已并入主仓），容器 pip 副本与真源
`cmp` **逐字节一致**（`ENV-COMPAT-IN-SYNC (944 lines)`）。**本波零 compat 补丁实付**
（无 `compat_patch_E.h`）——两件主判决在 v3 现成消费面全部编过。

---

## 0. 结论一览

| # | 条目 | 类型 | 触发形态 | 处置 | 取证 |
|---|---|---|---|---|---|
| G-E1 | **标量面三角件 `sinf/cosf` 不存在**（C 波 G-C1 的清单未列此二符号、`probe_math.py` 23 例亦无 sin/cos 用例——本波**补登记**） | compat 缺口（新登记，零补丁） | `T.cos/T.sin`(fp32) → codegen `cosf/sinf`（`intrin_rule_ascend.cc:22-38` AscendMath float32→`name+'f'`，`:56-57` REGISTER sin/cos） | **不补，判定绕开**：rope 走预计算 cos/sin 表 GM 加载（生产 rope_asc/rope_ref 契约形态）；取舍论证见 `RESULT.md` §③-1 | `verdict_log.txt` E-MATRIX rope sinf 段：`tl_kernel.asc:23:13: error: use of undeclared identifier 'cosf'` + 发射片段 `c = cosf(ang); sn = sinf(ang);`（sinf 的报错本体被 ASC 插件吞掉——`Diagnostic info can not be reported correctly` 已知形态，发射体留痕为证） |
| G-E2 | **G-C8 触发条件的正向再证实**（收窄，非新增） | codegen/UB 布局观察 | `stage` 变体 **2 个被访问的 alloc_shared**（`sc_ub/ps_ub` 各 `(WIN=8,)` f32） | 无需处置：两者正常落 `buf_dyn_shmem` 动态池且**偏移不重叠**（`[0:8)` 与 `[8:16)` float），命名空句柄=无（`null_ubuf_handle_named: False`） | `gen_attnsw_stage.asc:6,30,34,38-39,44` + 判决输出 |
| G-E3 | 体检工具坑（登记以免后人误判 G-C8 复发） | 方法论 | 粗正则 `__ubuf__ T* … = (__ubuf__ T*)0` 会命中**样板行** `buf_dyn_shmem = (__ubuf__ uint8_t*)0;`（该行动态池基址惯用写法，非命名空句柄） | 正则改为排除 `buf_dyn_shmem`（`(?!buf_dyn_shmem)`，`e_attn_sw_910b.py::scan_source`）；本波踩过一次假阳并如实记录 | `verdict_log.txt` 首跑（旧正则）曾报 `null_ubuf_handle: True`，修正后 `False` |
| G-E4 | 标量面整除/取模与窗口比较的 codegen 折叠（无缺口，留痕） | codegen 观察 | `t=u//heads; h=u%heads`、`j < i-max(i-7,0)+1` | simplifier 折叠正确：`gen_rope_table.asc` 里 UPC=heads=8 时退化成 `t=block_idx, h=uc`（零运行时除法）；窗条件化简为 `j <= i`（在 j∈[0,8) 定义域内与原判据等价，逐值核对通过） | `gen_rope_table.asc:17-29`、`gen_attnsw_pure.asc:27` |

## 1. 数学面消费清单（本波两件实际吃到的符号）

| 语义件 | DSL 写法 | codegen 发射 | compat v3 供给方 | 状态 |
|---|---|---|---|---|
| rope 角度 | `Cos[t,j]`/`Sin[t,j]` GM 标量直读 | plain deref（`gen_rope_table.asc:22-23`） | 表在 host fp64→fp32 预计算（`e_golden.py::rope_tables` 同式） | ✅ 零 transcendental |
| attn softmax exp | `T.exp(x)`（fp32） | `expf(...)`（`gen_attnsw_pure.asc:48,64`） | §11 `tl910b_expf` 软件件（真源 724 行起；C 波定标 max_rel_err=1.103e-06） | ✅ 首个**生产形态消费例**（C 波在 GDN SiLU 首用，本波落到 attn 对数域外路径） |
| 行最大 | `T.max(m, sc)` | `max(...)` | 原生标量 | ✅ |
| 归一除法 | `acc / l` | `/` | 原生标量 | ✅ 分母恒 ≥1（自见格 exp(0)=1），无需 FLOOR |
| 哨兵下溢 | `expf(-1e30 - m)` | 软件件早退分支 `x<=-104 → 0` | §11 | ✅ 结构性零权重（golden R5 按位验证） |

## 2. 与 B/C 波的关系

- **不新增任何补丁块** → 与 B/C 的合并冲突面为零（compat_gap_C.md §3 的枚举重定义之争与本波无关）。
- C 波遗留清单里与本波相关的两项，状态更新：
  - G-C1「剩余数学件」：`sinf/cosf` 由本波 G-E1 补登记（且带真实 kernel 形态的 FAIL 取证）；
    `logf/log2f/tanhf/powf/floorf` 仍缺、本波不需要（rope 表加载、attn 只吃 expf）。
  - G-C8：本波 G-E2 把「被访问 shared ≥2 → 正常分块」从推论升级为**实测再证实**（不同 kernel、不同缓冲宽度）。
- 单位/stride 证真（G-C4/G-C7）：本波两件**零 MTE 参与**（scalar 直读直写 + UB 元素级标量读写，无 `T.copy`/`asc_copy_*`），不新增证据也不依赖未证真面。

## 3. 判决

**主判决路无新 compat 缺口、零补丁实付**；唯一新登记 G-E1 是「确认缺 + 论证绕开」，
若编排要补软件三角件，按 C 波「软件件 + golden 逐行复刻 + 一符号一退让开关」模式即可
（范围规约必须做：rope 角 θ=t·freq 无上界，泰勒直接爆——`--mode rational` 件即该反例的骨架，
其数值仅在 |x|≲π 成立且**不进任何对拍判据**）。
