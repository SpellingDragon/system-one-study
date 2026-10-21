# compat_gap_B.md — 910B（dav-2201）port910b_compat.h 缺口清单与补丁片段

执行代理 B / p2-13-ascend-runtime P1-1。容器 `cann910b-b`，CANN 8.5.0 + bisheng(clang) 15.0.5，
tilelang pip 0.1.15，`-DTL_PORT910B_NATIVE_TYPES -I<asc/impl> -I<asc/include>` 注入态。

**主仓 `/tilelang/src/tl_templates/ascend/port910b_compat.h` 一字未改**（硬边界）。所有补丁只打在
容器内 pip 运行副本 `/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`，
由 `attempts/B/apply_compat_gapB.py` 幂等执行（原文件备份 `.preGapB`，锚点缺失即 FATAL 不改写）。

补丁总规模：**+3067 字节**（`apply_compat_gapB.py` 实测输出）。

## 0. 结论一览

| # | 缺口符号 | 触发形态（tilelang 生成码） | 原 compat 状态 | 处置 | 依据的 910B 原生件 |
|---|---|---|---|---|---|
| G1a | `asc_copy_gm2ub_align` | `asc_copy_gm2ub_align(ub, gm, 1, 8192, 0,0,0, static_cast<asc_load_l2_cache_mode>(0), 8192, 8192)` | **声明存在但实现错 arch**（3 参 `config` 形态只在 `__DAV_L310__/M310/L311` 段） | 重写为 7 参映射，bytes→32B 块 | `__cce_scalar::copy_gm_to_ubuf`（dav_c220 实现 :51） |
| G1b | `asc_copy_ub2gm_align` | `asc_copy_ub2gm_align(gm, ub, 1, 4096, static_cast<asc_store_l2_cache_mode>(4), 4096, 4096)` = **7 参** | 原 compat 声明 **10 参**（与 codegen 实发不符）+ 实现同 G1a 错 arch | 重写为 7 参签名 + 7 参原生映射 | `__cce_scalar::copy_ubuf_to_gm`（dav_c220 实现 :124/:221） |
| G2 | `asc_sync_notify` / `asc_sync_wait` | `asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0))`；`(PIPE_S, PIPE_MTE3, ...)` | **完全不存在**（`undeclared identifier`） | 新增**函数形宏**转 set_flag/wait_flag | `__cce_scalar::set_flag` / `wait_flag`（cce_aicore_intrinsics.h:2072/2742） |
| G3 | `asc_store_l2_cache_mode`（类型） | `static_cast<asc_store_l2_cache_mode>(4)` | **类型未定义**（只有 load 侧枚举） | 新增 enum（值域含 codegen 用的 4） | 无对应原生件：910B GM←UB 件不带 L2 ctrl 形参，该值在本层被丢弃 |
| G4 | `rsqrtf`（标量） | `rs = rsqrtf((acc / 512.0f) + 1e-5f)` | **不存在**（`undeclared identifier`） | 新增 `1.0f / sqrt(x)` shim | 标量 `sqrt(float)` 实测可用；见 §4 数值口径 |
| — | `expf` / `exp` / `log` / `logf` / `powf` / `fabs` / `floorf` / `ceilf` / `tanhf` / `sqrtf` | `T.exp` 落码为标量 `expf(...)` | 全部不存在 | **未补**（需裁决，见 §5） | 只有向量件 `__cce_scalar::vexp/vrsqrt`（需 VF 块，910B 外面 `T.Parallel` 被语义检查拒） |
| — | bf16 标量转换 | `(float)a[i]`，`a` 为 `__ubuf__ bfloat16_t*` | 与 compat 无关 | **未补**（bisheng 后端缺陷，见 §6） | `typedef __bf16 bfloat16_t`（__clang_cce_types.h:33）→ 内建类型，源码层无法重载转换 |

取证脚本（全部在 `attempts/B/`，可复跑）：`probe_910b_faces.py`（11 项 DSL 构件判决）、
`probe_setflag.sh`（G2 五种写法 A/B）、`probe_math_symbols.sh`（数学件一符号一 TU）、
`probe_cce_arch.sh`（dav-2201 预定义宏）、`dump_p8.py` / `dump_bf16.py`（位宽面取证）。

---

## 1. GAP-B 补丁片段全文

> 下面就是 `apply_compat_gapB.py` 里 `GAPB` 变量的原文。它被插在 compat 的
> `// codegen: asc_copy_gm2ub_align(...)` 起、到原 `copy_ubuf_to_gm(dst, src, config); }` 止
> 的位置（即**整段替换**原 10b 段的两个搬运件），仍在
> `#endif // TL_PORT910B_NATIVE_TYPES (AIC adapters)` 之前。

```cpp
// ── 10b-GAPB（P1-1 执行代理 B 于 cann910b-b 实测补齐；证据见 attempts/B/compat_gap_B.md）
#ifndef TL910B_GAPB_APPLIED
#define TL910B_GAPB_APPLIED

// G3: store 侧 L2 cache 模式枚举。发射点 codegen_ascend.cc:1430 以
// `static_cast<asc_store_l2_cache_mode>(n)` 传值（n=4 来自 copy.cc:316 kStoreDefaultL2CacheCtrl），
// 910B 的 GM<-UB 搬运原生件无 L2 ctrl 形参，本值在此层被丢弃。
enum asc_store_l2_cache_mode {
  ASC_STORE_L2_CACHE_ALLOC = 0,
  ASC_STORE_L2_CACHE_NORMAL = 1,
  ASC_STORE_L2_CACHE_FREE = 2,
  ASC_STORE_L2_CACHE_DEFAULT = 4
};

// G1: GM<->UB 搬运的 910B 正确形态（7 参原生件，长度/步长以 32B 为单位）。
// codegen 侧实发形态（src/ascend/codegen/codegen_ascend.cc:1405-1433 + src/ascend/op/copy.cc:300-328）：
//   asc_copy_gm2ub_align(dst, src, n_rows, row_bytes, 0, 0, 0, load_mode, src_stride_bytes, dst_stride_bytes)
//   asc_copy_ub2gm_align(dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)  // 7 参!
// 910B 原生件（asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h:51/124/221）：
//   copy_gm_to_ubuf(__ubuf__ void*, __gm__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   copy_ubuf_to_gm(__gm__  void*, __ubuf__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   —— 官方 :221 传 `tensorSize*sizeof(T)/32` 作 blockLen，故单位=32B；名字见
//      cce_aicore_intrinsics.h:992/1042（namespace __cce_scalar，无 arch guard）。
#define TL910B_BYTES_TO_BLK(x) ((uint16_t)(((int)(x)) / 32))

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                            int blockCount, int blockLenB, int rsvd0,
                                            int rsvd1 /*right_pad*/, int rsvd2,
                                            asc_load_l2_cache_mode mode, int srcStrideB,
                                            int dstStrideB) {
  (void)rsvd0; (void)rsvd1; (void)rsvd2; (void)mode;
  __cce_scalar::copy_gm_to_ubuf((__ubuf__ void *)dst, (__gm__ void *)src, (uint8_t)0,
                                (uint16_t)blockCount, TL910B_BYTES_TO_BLK(blockLenB),
                                TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src,
                                            int blockCount, int blockLenB,
                                            asc_store_l2_cache_mode mode, int dstStrideB,
                                            int srcStrideB) {
  (void)mode;
  __cce_scalar::copy_ubuf_to_gm((__gm__ void *)dst, (__ubuf__ void *)src, (uint8_t)0,
                                (uint16_t)blockCount, TL910B_BYTES_TO_BLK(blockLenB),
                                TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
}

// G2: 跨 pipe 事件对。发射点：T.copy(gm->ub) 后 `asc_sync_notify(PIPE_MTE2, PIPE_S,
// static_cast<event_t>(0))`，首个消费点 asc_sync_wait(...)；ub->gm 前为 (PIPE_S, PIPE_MTE3)
// （见 gen_vec.asc / gen_ub2gm.asc）。910B 依据：AscendC dav_c220 官方搬运即
//   SetFlag<HardEvent::MTE2_S>/S_MTE3 成对使用（kernel_operator_data_copy_impl.h:167-168,
//   kernel_event.h:57 MTE2_S）；toolchain 自身样板 __clang_cce_aicore_functions.h:2152-2159。
// **必须用宏、不能用 inline 函数**：__cce_scalar::set_flag/wait_flag 是
//   `__attribute__((clang_builtin_alias(...))) void set_flag(...)`（cce_aicore_intrinsics.h:2072/2742），
//   CCE 前端校验要求 pipe 实参为字面枚举常量；实测（attempts/B/probe_setflag.sh）：
//     V1 字面量直调        COMPILE-OK
//     V2 inline 函数转调   COMPILE-FAIL "the 1st parameter maybe need a type 'pipe_t'"
//     V3/V4 宏（带/不带括号）COMPILE-OK
//     V5 static_cast<event_t>(3) COMPILE-OK（event 可非 0）
//   这也解释了 toolchain 的 __cce_set_flag(:2796) 为何 (void)p;(void)tp 丢弃实参、硬编码
//   PIPE_M->PIPE_V —— 它无法转发变量形 pipe 实参。我们靠宏保住真实的 MTE2->S / S->MTE3 语义。
#ifndef asc_sync_notify
#define asc_sync_notify(from, to, evt) __cce_scalar::set_flag(from, to, evt)
#endif
#ifndef asc_sync_wait
#define asc_sync_wait(from, to, evt) __cce_scalar::wait_flag(from, to, evt)
#endif

// G4: 910B(dav-2201) **标量面没有 rsqrtf**。一符号一 TU 实测（attempts/B/probe_math_symbols.sh）：
//     sqrt(float)=OK；rsqrtf / sqrtf / expf / exp / logf / log / powf / fabs / floorf /
//     ceilf / tanhf / __nv_rsqrtf / __nv_expf 全 MISSING；__builtin_sqrt FAIL（aicore 函数禁 double）。
// tilelang 的 T.rsqrt 在 vector 面落码为标量 `rsqrtf(float)`（见 gen_rsqrt.asc），故补最小 shim。
// 数值口径：1/sqrt(x)，走已验证的标量 sqrt；**不等价于 RCRS/RSQRT 近似指令**，
//   编译面等价、上卡后与 AscendC 原生 rsqrt 可能有 ~ulp 级差异（RESULT.md 已申报）。
__aicore__ inline float rsqrtf(float x) { return 1.0f / sqrt(x); }

#endif // TL910B_GAPB_APPLIED
```

应用后的自检输出（`bash attempts/B/apply_compat_gapB.py` 形态，见 RESULT.md ①/2）：

```
PATCHED .../port910b_compat.h  (+3067 bytes)
  present asc_sync_notify:3  asc_sync_wait:3  asc_store_l2_cache_mode:3
  present asc_copy_gm2ub_align:2  asc_copy_ub2gm_align:2
  present #define asc_sync_notify:1  __cce_scalar::copy_ubuf_to_gm:1  float rsqrtf:1
```

---

## 2. G1 详证：GM↔UB 搬运件的 arch 错位

* **原状态**：compat 10b 段的 `asc_copy_gm2ub_align/asc_copy_ub2gm_align` 转调
  `copy_gm_to_ubuf(dst, src, config)` / `copy_ubuf_to_gm(dst, src, config)`（3 参 `config` 形态）。
  该形态在 `asc/impl/basic_api/` 里**只存在于 `__DAV_L310__ / M310 / L311` 段**；
  910B 编到 `__DAV_C220_CUBE__/__DAV_C220_VEC__`（`bisheng -E -dM --npu-arch=dav-2201` 实证，见
  `probe_cce_arch.sh`），910B 面**没有 3 参形态**。
* **为什么过去没暴露**：`e2e_cube.py`（E2E-CUBE-ONLY PASS）走的是 Cube 面 GM→L1
  （`copy_gm_to_cbuf`），从不触发 GM↔UB 路径 → compat 的 AIC/AIV 段搬运件此前**未经真实编译**。
* **codegen 实发参数序**（权威：`src/ascend/codegen/codegen_ascend.cc:1383-1433`，
  长度/步长换算见 `src/ascend/op/copy.cc:300-328`，`builtin.h:211` 的 8 参 IR 语义表）：
  * gm→ub：10 参 `(dst, src, blockCount, blockLenB, 0, 0, 0, load_mode, srcStrideB, dstStrideB)`
    → 原 compat 把第 3 参当 `sid` 用是**命名/语义错位**，实发是 blockCount。
  * ub→gm：**7 参** `(dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)`，
    第 5 参以 `static_cast<asc_store_l2_cache_mode>(4)` 发射 → 这也是 G3 的由来。
* **910B 原生件签名**（`asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h`）：
  ```cpp
  // :51
  __aicore__ inline void copy_gm_to_ubuf(__ubuf__ void *dstPtr, __gm__ void *gmPtr,
                                         uint8_t sid, uint16_t blockCount, uint16_t blockLen,
                                         uint16_t srcStride, uint16_t dstStride) { ... }
  // :124 / :221（:221 调用点传 `tensorSize * sizeof(T) / 32` 作 blockLen ⇒ 长度单位 = 32B）
  __aicore__ inline void copy_ubuf_to_gm(__gm__ void *gmPtr, __ubuf__ void *srcPtr, ...)
  ```
  名字面声明：`cce_aicore_intrinsics.h:992`（copy_gm_to_ubuf）、`:1042`（copy_ubuf_to_gm），
  均在 `namespace __cce_scalar`（`:496` 起，**无 arch guard**，故显式限定名调用）。
* **真实生成码**（`attempts/B/gen_addln.asc:12-13`，GAP-B 后编译通过）：
  ```cpp
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[8192]))),
                       (__gm__ uint8_t*)((&(X[((int32_t)block_idx) * 4096)])),
                       1, 8192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8192, 8192);
  ```
  → `blockLenB=8192`（=BM*DIM*2 字节）÷32 = 256 块，`blockCount=1`，与原生件语义一致。

## 3. G2 详证：跨 pipe 事件必须是字面量

* **原状态**：compat 完全没有 `asc_sync_notify/asc_sync_wait` → `use of undeclared identifier`。
* **A/B 取证**（`probe_setflag.sh`，同一 TU 模板、tilelang 同款编译 flag）：
  ```
  ===== V1_literal_all =====        RESULT: COMPILE-OK     # 字面量直调 __cce_scalar::set_flag
  ===== V2_wrapper_func =====       error: the 1st parameter maybe need a type 'pipe_t'
                                    RESULT: COMPILE-FAIL   # inline 函数转发变量形参 ⇒ FAIL
  ===== V3_macro_literals =====     RESULT: COMPILE-OK     # 宏透传字面量
  ===== V4_macro_parenthesized ===== RESULT: COMPILE-OK    # 宏 + 实参加括号仍 OK
  ===== V5_nonzero_evt_static_cast == RESULT: COMPILE-OK   # static_cast<event_t>(3) OK
  ```
  ⇒ **GAP-B 第一版写成 inline 函数是错的**（我踩了这坑并改回来了）：
  `__cce_scalar::set_flag(...)` 是 varargs builtin alias，CCE 前端只认字面枚举常量。
  这也解释了 toolchain 的 `__cce_set_flag(pipe_t p, pipe_t tp, event_t n)`
  （`__clang_cce_aicore_functions.h:2796-2810`）为什么 `(void)p; (void)tp;` 丢弃两个 pipe 实参、
  硬编码 `PIPE_M -> PIPE_V`：**它转不动变量形 pipe 实参**。
* **因此明确：不用 `__cce_set_flag/__cce_wait_flag`**（语义错误：会把 MTE2→S 的 DMA 完成信号
  换成 M→V），改用宏保真转发。add_ln 生成码里的实际事件对（`gen_addln.asc:14-22`）：
  ```cpp
  asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(1));   // Res 批量搬运完成
  asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));   // G/Bt 搬运完成
  ... asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(1)); // 首次消费点
  ```
  两个不同 event id 并存 ⇒ 宏转发形态能逐字保真（inline 函数版连编都编不过）。
* **注意 UB 侧写回**：`write_gm_bypass_dcache` 走的是 tl 模板的 dcache_bypass 标量写，
  **不产生 `asc_copy_ub2gm_align`**（本次 add_ln/readout 的真实码里都没有 ub2gm 调用）；
  G1b 的 7 参实现只在 `T.copy(ub, gm)` 形态（probe P2/P5）里被触发，那两个 case 已 PASS，
  但**上卡后的数值正确性仍未验证**（本地无 NPU）。

## 4. G4 详证：数学件面（rsqrtf 补，expf 不补）

`probe_math_symbols.sh`（一符号一 TU；批量 TU 会被 bisheng 的
`Fatal Err ... Diagnostic info can not be reported correctly` 截断，第一版因此漏判）：

```
OK      :: sqrt          ← 唯一可用的标量浮点数学件
OK      :: rsqrtf         ← GAP-B(G4) 打上后转 OK（补丁生效证据）
OK      :: rsqrt_alt      ← 1.0f/sqrt(x) 形态本身可用
MISSING :: rsqrtf(GAP-B 前) sqrtf expf exp logf log powf fabs floorf ceilf tanhf
MISSING :: __nv_rsqrtf __nv_expf
MISSING :: __builtin_sqrt → cast to/from double precision floating variable is not allowed in aicore function
```

* **rsqrt 数值口径**：`T.rsqrt` 落码是标量 `rsqrtf(float)`。GAP-B 用 `1.0f/sqrt(x)`。
  与 AscendC 的 `vrsqrt`（向量近似件）**不是同一条指令路径**，上卡后与 fp32 参考可能有
  ulp 级差异；LN 的 rsqrt 输入恒正（`var+eps>0`），`x<0` 不出现，故 `sqrt` 语义安全。
* **exp 不补的理由**：只有向量件 `__cce_scalar::vexp`（需在 VF/T.Parallel 块里操作 UB 张量），
  而 tilelang 的 `T.exp` 在标量域落 `expf`。要在 compat 里做一个"标量壳 + 内部起向量件"
  需要自配 UB 临时张量与 `pipe_barrier`，属于**性能/数值语义双向不可控**的改动，
  且本地无法对拍 → 留给编排者裁决（§5），不做未验证的绕行。

## 5. 未补缺口（需编排者裁决）

1. **标量 `expf/exp/log/logf/powf/fabs/floorf/ceilf/tanhf/sqrtf` 全缺**
   → 影响：readout 的 softmax/probs、任何含激活函数（gelu/silu）或 abs/floor 的 vector 面算子。
   候选方案：① probs 留 host；② 开 VF 块用 `__cce_scalar::vexp`（需先解决
   `T.Parallel must be inside a VF block` 的方言写法：910B 面如何发射 VF 块）；
   ③ compat 里做多项式近似（需数值验收口径，风险最高）。
2. **bf16 标量转换在 910B 后端不支持**：
   `fatal error: error in backend: not support bf16 type cast`（`dump_bf16.py` / `dump_p8.py` 取证，
   `bf16_in_f32_out / bf16_in_b16_out / f32acc_bf16_out / bf16_ub_copy_out` 四形态全崩，
   而 `fp16_in_fp16_out / f32acc_fp16_out_2outlets` PASS）。`bfloat16_t = __bf16`（内建类型），
   源码层无法重载转换 ⇒ **compat 补不了**，要绕只能：① 位宽口径钉 fp16（本次两 kernel 采用，
   且与生产一致）；② 以 uint16 搬运 + 手工位拼接（引入未验证的 bit 操作）；③ 走 Cube 面
   （bf16 在 L1/l1 路径是否可用需另测）。
3. **GAP-B 的搬运件有 32B 粒度约束**：`TL910B_BYTES_TO_BLK(x)` 是整数除法，
   非 32B 倍数会被**静默截断**（例：readout 里 26 个 int32 = 104B ⇒ 只搬 96B）。
   910B DMA 本身确以 32B 为块粒度，所以这不是补丁缺陷而是**契约**，但契约必须显式：
   候选 ① codegen/host 侧补 pad；② shim 内 `if (bytes % 32) 走标量逐元素兜底`；③ 向上抛编译期错误。
   本次两 kernel 已避开（add_ln 的搬运长度 8192/16384/2048B 全是 32 倍数；
   readout 全用标量散读不做批量 copy）。
4. **`AscendBinaryCache` 的 key 不含模板头文件内容**（`tilelang/ascend/backend.py:24-38`）：
   compat 改动后旧缓存仍会命中 ⇒ 判决必须像 `dump_final.py` 那样清缓存/强制 miss，
   否则"改了 compat 但编译结果没变"会被误读成补丁无效（我第一版就空手而归了一次）。

## 6. 环境侧附带缺陷（非 compat，但同类注入）

`release/ascend/port910b/bundle_ondemand.sh` 的 [3] 段（以及主仓之外的同款一行）
对 **tilelang 0.1.15** 的 `contrib/bisheng.py` 注入会产出 **SyntaxError**：
锚 `'"-O2", "-fPIC", "-std=c++20"'` 在该版本不是列表结尾（后面还有 `-mllvm
-cce-aicore-dcpreload-args=false`），而注入串硬加了 `"]`。
`attempts/B/fix_bisheng.py` 做了「撤销坏注入 → 正确追加列表元素 → `ast.parse` 自证」，
结果行（本容器实测）：

```
result = ["-O2", "-fPIC", "-std=c++20", "-DTL_PORT910B_NATIVE_TYPES",
          "-I/usr/local/Ascend/cann/aarch64-linux/asc/impl",
          "-I/usr/local/Ascend/cann/aarch64-linux/asc/include",
          "-mllvm", "-cce-aicore-dcpreload-args=false"]
```

这是**通用缺陷**（不是我这台容器的偶发），会影响一键上机；建议编排者把上层脚本改成
"在锚后追加列表元素、不闭合 `]`"，或直接复用 fix_bisheng.py。
