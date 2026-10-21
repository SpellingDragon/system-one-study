# compat_gap_C.md — 910B（dav-2201）port910b_compat.h 缺口清单与补丁片段（GDN 短卷积波）

执行代理 C / p2-13-ascend-runtime 甲路 P1-1。容器 `cann910b-c`，CANN 8.5.0 + bisheng(clang) 15.0.5，
pip tilelang 0.1.15，注入态 `-DTL_PORT910B_NATIVE_TYPES -I<asc/impl> -I<asc/include>`，
`ASCEND_NPU_ARCH=dav-2201 target_format=aibin`（与 tilelang codegen 同一条编译通路）。

**主仓真源一字未改**（硬边界）：`/tilelang/src/tl_templates/ascend/port910b_compat.h`
= **623 行 / 26895 字节**，mtime `Oct 8 11:38`（早于本会话），本波全程只读。
所有补丁只打在容器内 pip 运行副本
`/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`，
由 `attempts/C/env_setup.sh` 幂等装配（**每次**先 `cp -f /tilelang/.../*.h` 重置为真源，再重打）：

| 阶段 | 动作 | 副本行数 |
|---|---|---|
| [2] overlay | pip 副本 ← 主仓真源 | 623 |
| [2b] GAP-C | 追加 `compat_patch_C.h` 两块（自带守卫，不动 §0-§10 任何行） | 623 → **778** |
| [2c] §10b 就地改写 | `patch_compat_10b.py` 替换 `asc_copy_gm2ub_align` 体内实现（签名不变） | 778 → **799** |
| [3] bisheng | `patch_bisheng.py` 注入 -D/-I 锚 `'"-O2", "-fPIC", "-std=c++20"'` | — |

复跑：`bash attempts/C/run.sh --matrix`（4/4 `C-GDN-CONV-COMPILE-PASS`，见 `verdict_log.txt`）。

---

## 0. 结论一览

| # | 缺口 | 触发形态（tilelang 生成码 / DSL 写法） | 原 compat 状态 | 处置 | 910B 原生依据 / 取证 |
|---|---|---|---|---|---|
| G-C0 | `tl910b_bit_cast`（§1）**无可调用的 `__aicore__` 入口** | 任何 `__aicore__` 函数里调 compat §1/§5 的 helper | 声明为**无 `__aicore__` 限定**的 `inline constexpr` 模板 → CCE 重载集判 `no matching function for call to 'tl910b_bit_cast'`，且 ASC 插件把真实诊断吞掉只留 `Fatal Err` | **本块绕开**：位重解释就地写在 `__aicore__` 函数体内（`tl910b_gap_f2u/u2f`） | `probe_math.py` 逐符号 TU；诊断还原靠"去掉调用即 PASS"的差分 |
| G-C1 | **标量数学面整条不存在**：`expf sqrtf rsqrtf fabsf floorf logf log2f tanhf powf exp10f` | `T.exp`→`expf`、`T.abs`→`fabsf`（`intrin_rule_ascend.cc` AscendMath 对 float32 一律 `name+'f'`） | 全部**未声明**（`use of undeclared identifier`）；`<math.h>` 会把 host libstdc++ 拖进来撞 `stl_iterator.h` → 禁路；`__builtin_expf/_exp2f/_sqrtf/_floorf/_ceilf/_log2f` **编得过但 ld.lld undefined symbol**（libcall 无实现） | **补 2 个**：`tl910b_expf`（纯软件：范围规约 + 6 阶 Taylor + 指数域拼 `2^n`）与 `tl910b_fabsf`（让位 `__builtin_fabsf`，纯指令无 libcall），以**函数式宏**接无修饰全局名；退让开关 `-DTL_PORT910B_SKIP_expf` / `_fabsf` | `probe_math.py`（23 例，`PROBE-SUMMARY fails=18`）；数值精度实测见 `c_gdn_golden.py`：`EXP-SWEEP \|x\|<=30 max_rel_err=1.103e-06` |
| G-C2 | `asc_store_l2_cache_mode` 类型未定义 | codegen 发射 `static_cast<asc_store_l2_cache_mode>(4)`（`codegen_ascend.cc:1430`，值来自 `copy.cc:316 kStoreDefaultL2CacheCtrl=4`） | §10b 只定义了 **load** 侧枚举（`port910b_compat.h:571`），store 侧枚举**从未被定义过** → `unknown type name` | 新增 `enum asc_store_l2_cache_mode : unsigned char {…NORMAL=4}`；**固定底层类型**是必须的（C++11 fixed underlying type 才允许 `static_cast` 到 4） | `probe_dma.py::asc_store_l2_cache_mode_exists` PASS |
| G-C3 | `asc_copy_ub2gm_align` 签名与 codegen 实发 arity 不符 | codegen 实发 **7 参**：`(dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)`（IR 是 8 参，arg2=`sid` 被 `EmitUbufToGmCopy_` 主动丢弃） | §10b 声明 **10 参** → 该发射体**从未可编** | 新增 **7 参重载**（与 10 参原行共存，签名不删，保主仓兼容） | `probe_dma.py::codegen_ub2gm_7arg_as_emitted` PASS；负形对照 `arity_control_ub2gm_2arg/9arg`、`gm2ub_5arg` 分别报 `parameters too few/many` ⇒ **CCE 真校验 arity**，上面的 PASS 是强证据 |
| G-C4 | **MTE 长度/步长单位错位** | `asc_copy_*_align(..., 2048, 2048)` | §10b 把形参当"字节"直接塞进 packed config | 统一按 `bytes / 32`（`ONE_BLK_SIZE`）换算后再喂原生件（GAP-C 重载用 `>>5`，§10b 改写用 `/32`） | 单位证据：`asc/include/adv_api/matmul/matmul_client.h:2745` `repeatParams.blockLen = … * sizeof(T) / ONE_BLK_SIZE`；字节约定证据：`src/ascend/op/copy.cc:240-245 AscendMTEBytesFromElements`。**不换算=64 倍过搬。定性有据，数值仍待上卡证真** |
| G-C5 | §10b `asc_copy_gm2ub_align` **槽位错位 + 自造 config 位段错 arch** | codegen `EmitGmToUbCopy_`（`codegen_ascend.cc:1407`，`builtin.h:189` 文档）实发 10 参顺序为 `(dst,src,nBurst,burstLen,leftPad,rightPad,dataSelect,l2Ctl,burstSrcStride,burstDstStride)`；而 §10b 形参名写成 `(dst,src,sid,lenBurst,srcStride,dstStride,blockGroup,mode,nBurst,totalLen)` ⇒ sid 槽收 nBurst、nBurst 槽收 srcStride、totalLen 槽收 dstStride；体内自造位段 `sid\|nBurst<<4\|lenBurst<<16\|srcStride<<32\|dstStride<<48` 也和 950 `align_v2` 真位段（`cce_aicore_intrinsics_3101.h:1478`：`sid<<0\|burstNum<<4(21b)\|burstLen<<25(21b)\|leftPad<<46\|rightPad<<52\|dataSelect<<58\|l2Ctl<<60`；config1=`srcStride<<0(40b)\|dstStride<<40`）不符 | 同上 | `patch_compat_10b.py` **就地改写体内实现**：按 codegen 真实槽位取参 + 走 910B 原生 7 参 `copy_gm_to_ubuf(dst,src,sid,blockCount,blockLen,srcStride,dstStride)` + `/32`；padding/dataSelect 断言为 0（空宏占位）；**签名一字不动** | `ubstage` 变体 `C-GDN-CONV-COMPILE-PASS`（`gen_gdn_ubstage_exp.asc:24,33` 两条 MTE 真发射） |
| G-C6 | 910B 无 byte-granular burst：**非 32B 对齐 / 带 padding 的搬运未覆盖** | `leftPad/rightPad/dataSelect != 0`、`burst_len % 32 != 0`（GDN 常见 odd seq_len / 非 4 倍数通道块） | 无覆盖 | **未补**（本波只在 `X (C,L) f32, L=512` 即 2048B 对齐通路上立骨架）。需接 `align_b8/b16/b32` 变体 | `probe_dma.py` 覆盖面即对齐通路；缺口见 §4 |
| G-C7 | stride 语义 **gap（相对）vs absolute** 未定 | `burst_src_stride == burst_dst_stride == 行字节数` 时两者等价，无法区分 | 无说明 | **未决**，上卡对拍第一个要证真的点 | GDN 件里 `2048/2048`（见 `gen_gdn_ubstage_exp.asc`）正好同值 → 本波无法用编译面判别 |
| G-C8 | `T.alloc_shared` **命名空句柄**（codegen/UB 布局面，非 compat，但会静默吃掉语义） | kernel 里**被访问的 alloc_shared 恰好 1 个** ⇒ `__ubuf__ float *w_ub = (__ubuf__ float *)0;`（真实写 UB 偏移 0，且不进 `buf_dyn_shmem` 动态布局）；被访问的 shared ≥ 2 个 ⇒ 正常分块（偏移互不重叠） | — | **规矩**：纯标量件不留任何 alloc_shared（本波 scalar 变体已改成零 shared，`has_ubuf: False`）；需要 UB 就至少 2 个被访问缓冲或干脆走 `T.copy` 进动态池 | `probe_ubhandle.py` 4 case 判决：`CASE A/B/D refs=5~6 命名空句柄=(none)`、`CASE C refs=0 命名空句柄=['sa']` |
| G-C9 | 连续拷贝备选路不通 | `copy_data` / `copy_data_align64` | — | 只能走 `copy_gm_to_ubuf` / `copy_ubuf_to_gm`（7 参与 packed-config 3 参**两种形态都可编**，本波选 7 参） | `probe_dma.py`：`copy_data_gm2ub / copy_data_ub2gm / copy_data_align64_gm2ub` 全 FAIL `no matching function for call to 'copy_data'` |

> **无一处臆测**：G-C1/C2/C3/C8/C9 每条都在 `cann910b-c` 里以"一符号一 TU / 一形态一例"实测过，
> 并带**负形对照**（arity 三例）证明编译器确实在校验参数形态，PASS 不是假阳性。

---

## 1. GAP-C 补丁片段全文

### 1.1 块 1 — 标量数学面（`compat_patch_C.h` 前半，注入为 compat §11）

```cpp
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL_PORT910B_COMPAT_GAP_C_H
#define TL_PORT910B_COMPAT_GAP_C_H

// float <-> uint32 位视图（就地、__aicore__，不借 compat §1 模板 → 绕 G-C0）
__aicore__ inline unsigned int tl910b_gap_f2u(float x) {
  float q = x;
  return *reinterpret_cast<unsigned int *>(&q);
}
__aicore__ inline float tl910b_gap_u2f(unsigned int bits) {
  unsigned int u = bits;
  return *reinterpret_cast<float *>(&u);
}

// expf：910B 标量核无 SFU 入口 → 范围规约 + 6 阶多项式 + 指数域拼 2^n
__aicore__ inline float tl910b_expf(float x) {
  if (x <= -104.0f) return 0.0f;
  if (x >= 88.0f) return 3.4028234663852886e38f;
  const float kLog2e = 1.44269504088896340736f;
  const float kLn2 = 0.69314718055994530942f;
  const float t = x * kLog2e;
  int n = static_cast<int>(t >= 0.0f ? t + 0.5f : t - 0.5f);
  float r = x - static_cast<float>(n) * kLn2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666666666666f +
              r * (0.041666666666666664f + r * (0.008333333333333333f +
              r * 0.001388888888888889f)))));
  if (p < 1.0f) { p += p; n -= 1; }
  // p∈[1,2) 时 f2u(p) 的指数域**已是 127**，所以只能再叠 n（不是 n+127！）
  if (n < -126) return 0.0f;
  if (n > 127) return 3.4028234663852886e38f;
  const unsigned int mant = tl910b_gap_f2u(p);
  return (n < 0)
      ? tl910b_gap_u2f(mant - (static_cast<unsigned int>(-n) << 23))
      : tl910b_gap_u2f(mant + (static_cast<unsigned int>(n) << 23));
}

__aicore__ inline float tl910b_fabsf(float x) { return __builtin_fabsf(x); }

#ifndef TL_PORT910B_SKIP_expf
#define expf(x) tl910b_expf(x)      // 宏而非重载：上游补真声明也不会撞同名异种符号
#endif
#ifndef TL_PORT910B_SKIP_fabsf
#define fabsf(x) tl910b_fabsf(x)
#endif
#endif  // TL_PORT910B_COMPAT_GAP_C_H
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // TL_ASCEND_SIMT
```

### 1.2 块 2 — UB→GM 搬运面（G-C2 / G-C3 / G-C4）

```cpp
enum asc_store_l2_cache_mode : unsigned char {          // G-C2（fixed underlying type 必需）
  ASC_STORE_L2_CACHE_ALLOC = 0,
  ASC_STORE_L2_CACHE_NO_ALLOC = 1,
  ASC_STORE_L2_CACHE_NORMAL = 4                          // codegen 实发值
};

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src,
                                            int burst_num, int burst_len,
                                            asc_store_l2_cache_mode mode,
                                            int burst_dst_stride, int burst_src_stride) {
  (void)mode;                       // 910B 原生 7 参形态不吃 l2_cache_ctl（性能提示）
  copy_ubuf_to_gm(dst, src, (int8_t)0, (uint16_t)burst_num,     // sid：codegen 不发
                  (uint16_t)(burst_len >> 5),                   // G-C4 bytes -> 32B blocks
                  (uint16_t)(burst_src_stride >> 5),
                  (uint16_t)(burst_dst_stride >> 5));
}
```

### 1.3 §10b GM→UB 就地改写（`patch_compat_10b.py`，G-C5 / G-C4 / G-C6 占位）

签名保持与主仓 §10b 完全一致（10 参与形参名一字不动），只换体内实现，按 codegen
真实槽位取参并 `/32`；padding/dataSelect 走空宏 `ASCEND_910B_ASSERT_ALIGN32` 占位：

```cpp
const int burst_num = sid;              // 槽 3 实收 nBurst
const int burst_len_bytes = lenBurst;   // 槽 4
const int src_stride_bytes = nBurst;    // 槽 9 实收 burstSrcStride
const int dst_stride_bytes = totalLen;  // 槽 10 实收 burstDstStride
ASCEND_910B_ASSERT_ALIGN32(left_pad == 0 && right_pad == 0 && data_select == 0);
copy_gm_to_ubuf(dst, src, (int8_t)0, (uint16_t)burst_num,
                (uint16_t)(burst_len_bytes / 32), (uint16_t)(src_stride_bytes / 32),
                (uint16_t)(dst_stride_bytes / 32));
```

---

## 2. `expf` 双重加 127 的真 bug（无卡条件下唯一可行的抓法）

`tl910b_expf` 首版把指数域拼成 `n + 127`，而 `p` 已规格化在 `[1,2)`（其位模式的指数域
**本来就是 127**）→ 再加一次 127 等于 `2^127` 量级的错位：`exp(0)→1.7014e+38`、
`exp(1)→nan`、`exp(20)→-7.13e-31`（见 `attempts/C/_dbg.py`，本波留下的取证件）。

**没有 NPU 也能抓到它**的路径：`c_gdn_golden.py` 里的 `tl910b_expf()` 是 compat 的
**逐行 numpy 复刻**，与 float64 精确式对拍 → 复刻体一旦与手写 C 体同序，数值错位立刻现形。
⇒ 纪律：**compat 软件件与 golden 复刻体必须成对维护**，改一边就跑 `--golden`。
修复后实测：`EXP-SWEEP |x|<=30: max_rel_err=1.103e-06 OK`，
`EXP-EDGE expf(-200)=0 expf(200)=3.40282e+38 expf(-104)=0 expf(88)=3.40282e+38 expf(0)=1`。

## 3. 与执行代理 B 的补丁重叠（**合并前必须裁决**）

| 符号 | B（`attempts/B/apply_compat_gapB.py`） | C（本波） | 冲突 |
|---|---|---|---|
| `asc_store_l2_cache_mode` | 普通 `enum`，成员 `ALLOC=0 NORMAL=1 FREE=2 DEFAULT=4` | `enum : unsigned char`，成员 `ALLOC=0 NO_ALLOC=1 NORMAL=4` | **同名枚举重定义 → 两份一起注入必编译失败**；且成员语义命名不同（B 把 4 叫 DEFAULT，C 把 4 叫 NORMAL）。需统一（建议：fixed underlying type + 以 `copy.cc` 的 `kStoreDefaultL2CacheCtrl=4` 为唯一契约值，成员名对齐 AscendC 文档） |
| `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align` | **整段替换** §10b 两个适配器（7 参 + `TL910B_BYTES_TO_BLK(x)=x/32`） | gm2ub 就地替换体内实现（保签名）；ub2gm **另加 7 参重载**（10 参原行保留） | 两者单位换算结论**独立一致**（都对齐 /32，互相印证 G-C4）；但落法不同 → 同树注入会重签名。需选定一种 |
| `rsqrtf` | 补 `1.0f/sqrt(x)` shim（标量 `sqrt` 实测可用） | 未涉及（GDN 短卷积不吃 rsqrt） | 无冲突，建议上游采 B 的 shim |
| `asc_sync_notify/wait` | 补函数形宏转 `set_flag/wait_flag` | 未触及（本波无 Cube/mix 面，`has_asc_sync: False`） | 无冲突 |
| `expf` | **明确不补**，列为待裁决 | **补了软件 expf 并数值定标（1.1e-6）** | C 的实测把 B §5 的待裁决项变成"可采方案" |

## 4. 未补 / 需裁决清单（详见 RESULT.md §需裁决）

1. **G-C4/G-C7 数值证真**：单位 `/32` 与 stride gap-vs-absolute 只有上卡对拍能定论；本波已把"能编"与"数值对"分开登记。上卡前 **910B 首推 scalar 变体**（零 MTE，零歧义）。
2. **G-C6**：非 32B 对齐尾段 + padding/dataSelect 通路（`align_b8/b16/b32`）。GDN 生产 shape 若要支持 odd `L`/奇数通道块，必须补，属下一波。
3. **G-C0**：compat 是否补一个 `__aicore__` 版 bit_cast 入口（现每个补丁块都得就地重写位视图）。
4. **G-C1 剩余数学件**：`sqrtf/rsqrtf/logf/log2f/tanhf/powf/floorf` 仍未补。delta rule / softmax / RMSNorm 全都需要 → 建议一次性按本块模式立"910B 标量数学面"专章（软件实现 + `__builtin_` 优先 + 一符号一退让开关）。
5. **G-C8**：`(__ubuf__ T*)0` 单缓冲形态是否属 codegen bug（值不值得报上游）。
6. **§10b 改写是否回写主仓真源**：本波只改容器 pip 副本，真源仍是**已知不可编**状态（10 参签名 + 错 arch 位段 + 单位错），任何按真源 fresh 装的树都会挂 ubstage 路。
7. **环境侧附带缺陷（非 compat）**：`bundle_ondemand.sh [3]` 的 bisheng 注入片段本身会把 `bisheng.py` 打成 SyntaxError；`attempts/B/env_setup.sh` 内嵌同款片段（容器 b 同样风险）。本波的 `patch_bisheng.py`（锚 `'"-O2", "-fPIC", "-std=c++20"'` + 幂等 + 语法自检）建议 A/B/C 统一采用。

## 5. 取证脚本（全部可复跑，均在 `attempts/C/`）

- `probe_math.py` — 数学面 23 例（未声明 / 编得过链不上 / 原生可用 三分类），`PROBE-SUMMARY fails=18`
- `probe_dma.py` — 搬运面 19 例（含 3 例 arity **负形对照**），`PROBE-DMA-SUMMARY pass=13 fail=6`
- `probe_ubhandle.py` — UB 句柄形态 4 case（G-C8 触发条件）
- `c_gdn_conv_910b.py --dump` — codegen 产物（`gen_gdn_{scalar,ubstage}_{exp,rational}.asc` + `gen_gdn_scalar_pre_cleanup.asc` 为 G-C8 反例存档）
- `c_gdn_golden.py` — CPU 语义 golden（四路真值 + exp 精度扫描 + npz 产出），`GOLDEN-PASS`
- `env_setup.sh / patch_bisheng.py / patch_compat_10b.py / run.sh` — 装配与一键判决（`run.sh --matrix` 每次换新 `TILELANG_CACHE_DIR`，防陈旧 PASS）
