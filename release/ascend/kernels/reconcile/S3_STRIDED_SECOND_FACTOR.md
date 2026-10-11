# S3_STRIDED_SECOND_FACTOR.md — P1-1s 跨步装填「第二层因素」取证与判读

> 任务归属：p2-13 / 执行代理 **S3**。**不占卡**：本文件全部结论只引用**已归档的一手产物**
> 与**容器内一手头文件**，外加可复跑的离线算式。**不改 trunk**（只读）。
> 三态标注：✅实测原文 / ⚠️提及未取 / ❓推断。每条判读挂证据行号。

---

## 0. 一句话结论

1. **一手 ISA 证据（✅）**：910B 原生 `copy_gm_to_ubuf` 的第 6/7 参是**块间空隙 (gap)**，
   **不是**首地址间距 (pitch)。三处独立原文互相印证（§2）。⇒ 我方 compat 把 codegen 的
   **pitch** 直接灌进 gap 槽，硬件把有效步长算成 `blockLen + pitch`。**这是 H-c/H-e 的根**。
2. **第二窗「Z(dstStride 被忽略)」判决（⚠️）是度量伪影**：探针候选打分对 NaN 单边不罚
   （`...:215-216`），使"只预测少量有限格"的候选（Z 只提交 48 个格子）天然获胜。§4 给出算式。
3. **残留（45/48、640/1152、UNATTR 640/1152）不是任何单一 (sp,dp,off) 几何能复现的**：
   离线穷举 308,945 组（§5）零命中，最小 L1=167 ⇒ 第二层因素**至少含一个非几何成分**，
   最可能为 **H-b（w_ub 未清零 / stale）**，但**未直测 ⇒ 记「不可判（倾向真）」**。
4. **提案**：compat 层 pitch→gap 归一（`attempts/S3_compat_proposal.diff`，编译面已自证 PASS，§8）；
   若上卡后失配未归零，退到 plan 层"每突发单发连续 DMA"（§6）。

---

## 1. 复跑入口（每条都可复跑）

```bash
cd /Users/pengweiye/Documents/codes/system-one/release
# ① 两窗归档 md5 / 条目数（本窗复核，与任务书同值）
md5 runs/1011-p2-13-oncard-window2-throughput/artifacts/sys1_kernels.tgz   # 440b59589201c7545db3d8e7430c1f1c  431 条目
md5 runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz          # f2ba9a2080de73d1d04d4a309dc8496e  422 条目
# ② DMA 形态普查（见 §6 表；脚本 digest 已随本窗新建于 /tmp，逻辑=解析 asc_copy_gm2ub_align 10 实参）
# ③ 病档原文（k=48 fp32 多突发跨步）
tar -xzf runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz -C /tmp/s3_w1
cat -n /tmp/s3_w1/0.1.15/*/kernels/5d732a8f*/device_kernel.asc           # 第 31 行 = DMA
# ④ 一手头文件（容器 cann910b-h，CANN 8.5.0）
docker exec cann910b-h bash -lc 'sed -n "34,60p"  /usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/include/basic_api/kernel_struct_data_copy.h'
docker exec cann910b-h bash -lc 'sed -n "344,353p" /usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl/basic_api/utils/kernel_check_data_copy_util.h'
docker exec cann910b-h bash -lc 'sed -n "55,76p"  /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp'
# ⑤ 离线复算（本窗新建，不改产品语义）
.venv/bin/python ascend/kernels/reconcile/S3_probe_semantics_repro.py
# ⑥ 候选「越界数」逐档复算（对拍 w2_pitchgap.log:16/30/44 的`写/读越界`，本窗已复核逐值相等）
python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev cpu   # 或直接 import predict() 逐候选打表
```

---

## 2. 一手 ISA 证据（✅ 实测原文，三条独立互证）

**证据 A — 字段定义：stride 与 gap 是 union 别名（值域同一）**
`cann910b-h:/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/include/basic_api/kernel_struct_data_copy.h:34-58`
```cpp
struct DataCopyParams {
    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    // srcStride and dstStride will be deprecated, use srcGap and dstGap instead.
    union { uint16_t srcGap = 0; /* srcStride will be deprecated, use srcGap instead */ uint16_t srcStride; };
    union { uint16_t dstGap = 0; /* dstStride will be deprecated, use dstGap instead */ uint16_t dstStride; };
};
```
⇒ ✅ **`srcStride ≡ srcGap`、`dstStride ≡ dstGap`**（同一存储、同一语义），官方已把名字从 stride 迁到 gap。

**证据 B — 越界校验算式：总跨度 = 段长 + (n-1) 空隙**
`.../asc/impl/basic_api/utils/kernel_check_data_copy_util.h:346-351`
```cpp
srcMaxOffset = (uint64_t)nBurst * lenBurst * srcBurstLenUnit + (nBurst - 1) * srcStride * srcStrideUnit;
dstMaxOffset = (uint64_t)nBurst * lenBurst * dstBurstLenUnit + (nBurst - 1) * dstStride * dstStrideUnit;
```
⇒ ✅ 若 stride 是 **pitch**，跨度应为 `(n-1)*pitch + len`；此处是 `n*len + (n-1)*stride`
= **每块长度累加 + 每两块之间留 stride 空隙** ⇒ 语义 = **gap**。（单位 `srcStrideUnit` 由
`GetStrideUnit`(:303-312) 取，缺省 `DEFAULT_C0_SIZE`=32B 块，与 compat `TL910B_BYTES_TO_BLK` 一致。）

**证据 C — 官方 PTO(910B/a2a3) 的 gap 计算：gap = 行步长 − 行长**
`.../include/pto/npu/a2a3/TLoad.hpp:69-72`
```cpp
uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);   // gap = (行步长-行长)*元素字节
uint32_t gmGap = (uint32_t)gmGapValue;
uint32_t ubGapElement = (TileData::Cols - validCol);
uint32_t ubGap = (ubGapElement * sizeof(...DType)) >> SHIFT_BLOCK_BYTE;          // UB 侧 gap 折算成 32B 块
```
同文件 `:20-26` 把它喂给 `copy_gm_to_ubuf_align_b8/b16/b32(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap)`。
⇒ ✅ **官方模板自己就是 `gap = stride − len`**，与 codegen 直接传 stride(=pitch) 的做法**相反**。

**结论（H-c/H-e 的直接依据）**：codegen (`copy.cc:242-245/310-313` + `codegen_ascend.cc:1416-1417`)
按 **pitch** 计算 `plan_src_stride_bytes / plan_dst_stride_bytes`，compat (`port910b_compat.h:607-609`)
原样 `TL910B_BYTES_TO_BLK(...)` 灌进 gap 槽 ⇒ 硬件有效步长 = `blockLen + pitch`。k=48 实发
`(len=64B, src=192B, dst=64B)` ⇒ 硬件有效 **源 256B(=64 元素，应为 48)、目的 128B(=32 元素，应为 16)**。

---

## 3. H-a … H-e 逐条判读

| ID | 假设 | 判读 | 证据（行号） |
|---|---|---|---|
| **H-a** | 末块 `blockLen` 截断（尾突发不足整块） | **假** | 见下 |
| **H-b** | UB 未清零 / stale（= `UNATTRIBUTED` 的字面义） | **不可判（倾向真）** | 见下 |
| **H-c** | `srcStride`×`blockCount` 是"每突发加 gap"而非等步长 pitch | **真** | 见下 |
| **H-d** | compat 的夹址 `min()`/行号守卫在跨步档写错目标 | **假**（对现 trunk） | 见下 |
| **H-e** | `plan_single_row` 生成的参数组合本身不合法 | **真（弱形式）**；强形式"必须循环拆趟"=**不可判** | 见下 |

### H-a = 假（✅ 结构证据）
`k=48` 实发 `blockCount=32, blockLen=64B`（`5d732a8f.../device_kernel.asc:31`）；`bk=16` fp32
⇒ 每突发 64B = **2 个整 32B 块**；`32×64B = 2048B = w_ub` 区（`[0,512)` fp32，见 :40 消费
`w_ub[j_1*16+kk_1]` 与 :15 `c_ub=i*32+j+512`）。`n=96=3×32`、`m=16` 均整除 ⇒ **无尾突发、无边界截断**。
`k=96` 同（`a421afa3...:31` `32, 128`；`32×128B=4096B=1024 fp32`）。⇒ H-a 的"48 非 32 对齐"
前提不成立（`48×4=192B`、`16×4=64B` 皆 32B 对齐）。⚠️ 注：`copy.cc:273-276` 确有 `data_select`
右补齐路径，但那是另一条分支（`do_pad`），k=48 内核未走。

### H-b = 不可判（倾向真）❓——本任务核心"
证据链：
- **w_ub 未被初始化**：`5d732a8f...:12-48` 只初始化 `c_ub`(:13-17) 与 `a_ub`(:19-29)，**`w_ub[0,512)`
  在上卡前无任何清零**（对比 `a421afa3:13-17` 也只清 `c_ub`）。⇒ DMA 未写到的槽位保留**上一次发射
  的同地址内容**（stale）。
- **gap 语义会让 DMA 少写一半 w_ub**：若硬件按 gap 算（§2），`dstStride=64B` ⇒ 有效目的步长 128B
  = 32 元素 ⇒ 突发只在 `[0,512)` 的**偶 16-块**落数据，**奇 16-块空洞**；且 `r≥16` 的突发落到
  `[512,1024)`（**踩 c_ub**）。空洞恰是 stale 的宿主。
- **不变量反证**：实测 `UNATTRIBUTED=0`（k=48，全 512 槽有限）却 `OK=0`（全行错）——与"纯 DMA 几何"
  不相容（§5 穷举 0 命中），但**与"部分 DMA + 部分 stale 填满"相容**。
- k=96 的 `UNATTRIBUTED=640`（`w2_pitchgap.log:29`）比 k=48 大得多，且 **relu↔none 一变就跳**
  （:29 `640` → :36 `1152`）：两枚内核 UB 复用路径不同 ⇒ stale 内容不同 ⇒ 与"H-b 主导"一致。
**未直测 ⇒ 判「不可判」**；§7/§9 的下窗一枪（UB 预清零）即为其判决实验。

### H-c = 真（✅ 一手 ISA）
见 §2：stride 是 gap，`nBurst` 个突发的**源/目的指针每突发线性前移 `len + gap`**，即"等步长但步长含空隙"，
**不存在**"每突发重算源指针"之外的第三种解释。codegen 把 pitch 灌进 gap 槽 ⇒ 实际前移量被放大。

### H-d = 假（对现 trunk，✅ 代码证据）
compat `asc_copy_gm2ub_align` 全文（`port910b_compat.h:601-610`）**没有任何 `min()`/行号守卫**——
它是 10→7 参的**纯透传**。病档里唯一的 `min(...)` 在 **A 侧标量读**（`5d732a8f...:22-23`
`A[min(my*16+i_1, m-1)*48 + ko*16 + kk]`），属标量通道，**不触碰 W 的 DMA / w_ub 布局**。
⇒ "compat 夹址守卫在跨步档写错目标"这一具体机制在当前 trunk **不存在**。若 H-d 另指其它守卫，
请指认符号——本窗未发现。

### H-e = 真（弱形式，✅）／不可判（强形式，❓）
`copy.cc:286-289` 折叠判据：`n_rows==1 || (src_stride==row_bytes && dst_stride==row_bytes)`；
`:310-313` 非折叠支发 `blockCount=n_rows` + **pitch** 两个 stride。结合 §2 的 gap 语义 ⇒ 这条
"单次多突发 DMA"的参数组合**语义自相矛盾**（把 pitch 当 gap 用）⇒ **弱形式的『组合不合法』= 真**。
但**强形式**"910B 需要的是循环拆趟而非单条 DMA"**无法用归档+头文件判决**（要上卡）⇒ **不可判**。
不过修复上二者指向同一退路（§6 提案 B）。

---

## 4. 探针度量缺陷（⚠️：第二窗「Z 判决」为何不可信）

`U2_probe_k48_dma_semantics.py:211-218` 的候选打分：
```python
mm = ((take != mrow) & np.isfinite(mrow) & np.isfinite(take)).sum()
nn = (np.isfinite(take) & ~np.isfinite(mrow)).sum()
scores.append((tag, mm + nn, oob, doob))
```
`nn` 只统计**"候选有限 & 实测 NaN"**，**从不统计"实测有限 & 候选 NaN"**。⇒ 候选若把大片格子
预测成 NaN，这些格子**完全不进惩罚**。

**提交格数（U2 口径，✅ 本窗离线复算 `S3_probe_nextshot.py` 逐档实测）**：U2 只在
`take = pr[:min(16,bk)]`（≤16 行）里罚分，Z 因 `dp=0` ⇒ `dslot=c` ⇒ `j_1=c//bk=0`，**每个 (源行 mx)
只提交 1 列**，故 Z 在 16 行窗口内**只提交 48 格**（k=48/96/32 三档同值）；同口径下
`G/Gd=768`、`Gs=1408/1408/1280`、`P/C/Zc=1536`。⇒ 于是 `Z mismatch=45`(k=48,:16) / `48`(k=96,:30) /
`48`(k=32,:44) 的真实含义是 **"在它敢预测的 48 格里错 45/48/48 格"**，而非"1536 格里错 45"。
**"Z 远小于 P/G/C"是候选提交格数差异造成的伪影**，不能据此判"硬件忽略 dstStride"。
⇒ 亦解释 `U2-DMA-READOUT-DONE unexpected=4`（:54）。

**第二处缺陷（概念层）**：判决叙述说"硬件 **每突发按连续写**"
（`w2_pitchgap.log:18/25`），但候选 `Z(dstStride被忽略)` 的定义是 `(sp=k, dp=0)`
（`:189-191`，`predict` 里 `dslot=r*0+c`）——**"目的指针完全不前移"**，与"每突发连续写"
（那应是 `(sp=k, dp=bk)`，恰等于 `P`）**不是一回事**。⚠️ 即"叙述的机制"与"胜出的候选模型"
二者不自洽 ⇒ 双窗那份"已裁决事实"的**机制描述亦不可信**，须并入下窗复核（§7 附表末行）。

---

## 5. 最简可解释模型：**能否用一条统一规则解释 45/48 与 640/1152？—— 不能**

离线复算件 `S3_probe_semantics_repro.py` 把六候选与实测图对拍，并做**三维穷举**（源步长 `sp`、
目的步长 `dp`、全局源偏移 `soff`），要求复现实测四元组 `(OK, 行错位, 列错位, UNATTR)`：

| 目标档 | 实测四元组（`w2_pitchgap.log`） | 穷举范围 | 命中 | 最小 L1 |
|---|---|---|---|---|
| k=48/bk=16 | `(0, 1536, 1353, 0)` (:15) | `sp∈[0,96] × dp∈[0,48] × soff∈[-32,32]` = 308,945 组 | **0** | **167** |
| （含 `dp=bk` 特化，段命令另一轮） | `(0, 1536, 0)` | `sp∈[0,128] × soff∈[-96,96]` | **0** | — |

关键对照（同一复算件输出）：
- **P(pitch,pitch)=(48,16)**：`mismatch=1536`（六者中最差）—— codegen **本意**（pitch）被实测**全否**。
- **G(双gap)=(64,32)**：`mismatch=384`，但其预测图**有大片 NaN 空洞**（`写越界块=768`），
  与实测 `UNATTRIBUTED=0` **直接冲突** ⇒ 忠实 gap 语义**也不自洽**。
- **Z=(48,0)**：靠 §4 伪影"胜出"。

⇒ **不存在单一 (sp,dp,off) 线性几何**复现实测图（308,945 组零命中，最小残差 167）。
**唯一与全部残留相容的最简模型是"两因子复合"**（❓推断，待下窗证）：
1. **因子一（几何/规格）**：单次多突发 DMA 把 pitch 当 gap ⇒ 目的侧少写一半、源侧多跨，
   且越界写踩 `c_ub`（P1-1s 第一层，§2 一手坐实）。
2. **因子二（状态/脏读）**：`w_ub` 未清零，DMA 空洞由 **stale** 填充（H-b）⇒ 实测"全有限但全错行"，
   且 k=96/relu 一变就多 512 个 `UNATTRIBUTED`。
**保留意见**：因子二只是"最能解释零命中"的假设，**本窗无直测**；也可能存在第三因子（如
`c_ub`↔`w_ub` 双向踩踏）。故本节记 **❓推断**，不下"已定"。

---

## 6. 影响面 & 提案方向

**跨步（`blockCount>1`）件普查**（两窗全量解析 `asc_copy_gm2ub_align` 实参）：

| 归档 | 内核 | BC | blockLen | srcStride | dstStride | 身份 |
|---|---|---|---|---|---|---|
| w1 | `5d732a8f` | 32 | 64 | 192 | 64 | gemm fp32 n=96 **k=48**（病档） |
| w1 | `a421afa3` | 32 | 128 | 384 | 128 | gemm fp32 k=96（n=32） |
| w1 | `ea1059e6` | 32 | 64 | 192 | 64 | gemm fp16 k=96（n=32） |
| w1,w2 | `18502bd3` | 2048 | 256 | 4096 | 256 | bf16 matmul（`__mix__(1,2)`） |
| 其余 | 40+ 枚 | **1** | =stride | — | — | SINGLE（本改动恒等，无回归） |

⇒ 影响面 = **所有 `blockCount>1` 且 `srcStride≠blockLen` 的 GM↔UB 搬运件**（上表 4 枚）；SINGLE 档
`gap = pitch - blockLen = 0`，与旧码**逐字节等价**。

- **提案 A（本 diff，compat 层）**：`gap_bytes = pitch_bytes − blockLen_bytes`（`<0` 夹 0）。
  最小、可编译、对 SINGLE 无回归。**前提**：硬件忠实实现 gap 语义（§2 一手支持）。
- **提案 B（plan 层，未在本 diff）**：把跨步档**拆成每突发单发连续 DMA**（`blockCount=1`）或改标量循环
  ⇒ 两个 stride 槽**根本不参与** ⇒ **与 gap/pitch/dstStride-被忽略 三种语义全都无关**，是**语义无关的稳妥解**。
  代价：`bn=32` ⇒ 32 条 MTE2（指令数×32，性能面需评估）；或 `T.copy` 改循环由编译器流水。
  **何时退到 B**：下窗若 A 打上后失配**未归零**，说明"忽略 dstStride"成立 ⇒ A 不足 ⇒ 走 B。

---

## 7. 未判定项登记（诚实清单）

| 项 | 状态 | 缺什么 | 下窗一枪（§9）如何判 |
|---|---|---|---|
| H-b（UB stale） | ❓ 不可判（倾向真） | 一次"UB 预清零"对照 | 预清零后 `UNATTR` 是否归零 → §9 真值表 |
| H-e 强形式（是否必须拆趟） | ❓ 不可判 | 上卡比 A/B 两形态失配 | A 版失配是否归零 |
| 残留 640/1152 的成因分解 | ❓ 不可判 | 干净的"因子一已修"基线 | A 打上后的残差图 |
| （附带发现）"硬件忽略 dstStride"这一"已裁决事实" | ⚠️ 证据不成立 | —— | §4 已证其为**度量伪影**，请并入下窗复核 |

---

## 8. 编译面自证（✅，容器 `cann910b-h`，trunk 零改动）

```bash
# 副本打补丁（/dev/shm，盘紧）—— 绝不改 install/trunk 头
docker exec cann910b-h bash -lc '
  TPL=/usr/local/lib/python3.10/dist-packages/tilelang/src
  rm -rf /dev/shm/s3_inc && mkdir -p /dev/shm/s3_inc && cp -r "$TPL/tl_templates" /dev/shm/s3_inc/
  cd /dev/shm/s3_inc && patch -p2 < /tmp/S3_compat_proposal.diff'   # 打补丁后的 md5 = 6d93fcfec49c836bfddf75dca1d48542
# 编（补丁副本 / 原装对照）
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh>/dev/null 2>&1; \
  python3 /tmp/S3_probe_compile_patch.py --inc /dev/shm/s3_inc'     # → S3-COMPILE PASS
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh>/dev/null 2>&1; \
  python3 /tmp/S3_probe_compile_patch.py'                           # → S3-COMPILE PASS（对照）
```

实测输出：
```
S3-COMPILE-HEADER /dev/shm/s3_inc/tl_templates/ascend/port910b_compat.h md5=6d93fcfec49c836bfddf75dca1d48542
S3-COMPILE PASS inc=/dev/shm/s3_inc
S3-COMPILE-HEADER .../3rdparty/../src/tl_templates/ascend/port910b_compat.h md5=ca023c2e01746ca52c87f36bcea93ee9
S3-COMPILE PASS inc=<install>
```

**trunk 完整性复核（改前后同值，✅）**：
```
host tileslang/src/.../port910b_compat.h        ca023c2e01746ca52c87f36bcea93ee9   （改前/改后）
host system-one release/ascend/port910b/patches  ca023c2e01746ca52c87f36bcea93ee9   （改前/改后）
容器 install .../tilelang/src/.../port910b_compat.h ca023c2e01746ca52c87f36bcea93ee9 （改后）
```
补丁 `--dry-run` 对 trunk 亦 PASS。

---

## 9. 下窗一枪（`reconcile/S3_probe_nextshot.py`）

一条命令级探针，在 U2 探针**产品语义不变**的前提下，把三件事并到一枪：
1. **提案 patch 前/后**：同形状两条腿 —— 走 compat 现态（pitch 灌 gap）与"提案 A 现态"（gap 归一）；
   若 A 未落地，则用候选集里的 `A(GAP-A) = (k+bk, bk)` 与 `A2` 行等价标注（见脚本真值表）。
2. **H-b（UB 预清零）**：`--preclear` 开/关两态，看 `UNATTRIBUTED` 是否归零。
3. **度量修正**：新增**对称**失配 `sym = (实测有限 & 候选NaN)` 计入 → 破除 §4 伪影。

真值表（脚本内常量 `TRUTH`，随跑随打印）：

| 观测（A=提案A失配，U=UNATTR，sym 显示） | 指向 |
|---|---|
| A=0 且 U=0（preclear 与否皆然） | **因子一=根**（H-c/H-e 真）；H-b 无关；提案 A 充分 |
| A=0 仅当 `--preclear`；否则 A>0 | 因子一 + **H-b 并存**；A 需配"清零 w_ub" |
| A>0 且 preclear 不改 U | A 不足（"忽略 dstStride" 成立）⇒ **退提案 B（拆趟）** |
| A=0 但出现新写越界 | A 的 gap 归一有边界错（dst gap 应=0 时被误扩大）⇒ 复核 §2 |

---

## 10. 纪律自陈（本窗实际遵守 / 未达）

- 「不占卡」：全程只用归档产物 + 容器**编译**（未跑 kernel，未 init NPU 执行）。✅
- 「不改 trunk」：trunk 三处 md5 改前=改后=`ca023c2e…`（§8）。✅
- 「禁改清单」：未触碰 `*_asc.py/*_kernel.py/run_numerics.py/bench_card.py/D2_*/compile_all_asc.py/
  card_run.sh/runs/**/tasks.md`；未 `git`。✅
- 「区分三态」：§2=✅（一手头文件）/ §3-H-b、§5=❓/ §6-H-e 强形式=❓；"硬件忽略 dstStride"=⚠️（§4）。
- **最大风险自陈**：§5 的"两因子模型"是**推断**（零命中反证强，但因子二未直测）；已按 ❓ 标注，
  未写成事实。
