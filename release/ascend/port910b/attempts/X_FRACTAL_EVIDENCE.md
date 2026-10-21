# X_FRACTAL_EVIDENCE — p2-13 P1-1j：dav-2201(910B) cube intrinsic 分形/参数语义取证

变更：`teacher-p2-production-full / p2-13-ascend-runtime` → tasks.md **P1-1j**（performance-track 考古，D-cube1 第二轨）
代理：X（本文件作者）｜日期：2026-10-10｜容器：`cann910b-k`（CANN 8.5.0 aarch64，bisheng 15.0.5）
纪律：本任务**未改动任何产品代码**；证据全部来自容器内只读探索 + `attempts/X_probe_*.py` 新 TU 编译探针 + hivmc bitcode 反解。

---

## 0. 结论速览（四条裁决，置信度标注）

| # | 裁决 | 置信 |
|---|---|---|
| **①** | **2201 的 `load_cbuf_to_ca` 真实形 = V1 `LoadData2D` 族 9 参**：`(dst __ca__, src __cbuf__, startIndex u16, repeatTimes u8, srcStride u16, dstGap u16, sid u8∈[0,15], transpose **bool 字面量**, addrCalMode addr_cal_mode_t)`。任务书所怀疑的 3101 显式形 `(mStart,kStart,mStep,kStep,srcStride,dstStride,transpose)` 与 5 参 config 形，在 dav-2201 **均不被接受**（头文件级 + sema 级双证）。→ **"若 2201 实际接受前者则我们喂的是错位语义"这一支假设被否**：P1-1d 映射到 V1 族是对的，错在**槽位分配**（见③）。 | 高 |
| **②** | **GM→L1 必须产出 NZ（分形）布局**。官方 Matmul 写 L1 只有 `CopyND2NZ` / `CopyNZ2NZ` 两条路，**不存在"ND 线性直写 L1 再喂 cube"的路**；2201 的 ND→NZ 硬件件 = `copy_gm_to_cbuf_multi_nd2nz_b8/b16/b32s`（11 参，本窗实证可编）。§12 现用的 `copy_gm_to_cbuf` 8 参裸形与官方 `DataCopyGM2L1Impl`(纯 ND 块拷贝) **逐字同款** → 产出 ND，不是 NZ。 | 高 |
| **③** | §12 `asc_copy_l12l0a/b` 的**头号致命槽 = sid**：codegen 第 3 实参是 IR 的 `m_start`（row16 块起点，可 0..127），而 wrapper 形参把它命名为 `sid` 并塞进硬件 **4-bit** sid 槽；本窗实证 **运行期值不经 sema 区间检查**（RT3 rc=0），故 `m_start>15` 时静默产生非法 L1 分区号 → 与 `aicore exception 507015`（DMA 非法访问）症状直接吻合。其余错位见 §7 逐槽表。 | 高（因果链最后一寸仍需上卡 E-3 对照坐实） |
| **④** | **未定论项**（不臆造）：L0 目的侧 `dstGap`/每趟步进、`_transpose` 件的 cfg 位序、ND2NZ 与 `asc_fill_l1` pad 区的交互。已给"下窗一次做完"的最小判别实验设计（§8）。 | — |

**路径 2（gitee 官方低级 API 样例）：本窗未取证**——本 agent 无 WebFetch 工具，任务书纪律禁 `curl/wget` 下载。替代取证见 §5：**直接用容器内与编译器同版本的 CANN 8.5.0 AscendC 安装树**（`asc/impl/adv_api/detail/matmul/**` + `asc/include/interface/*.h` 文档注释），其权威性强于 gitee 样例（版本严格匹配 `bisheng 15.0.5 / CANN 8.5.0`）。

---

## 1. 复跑入口（全部命令可复跑）

```bash
# 探针脚本（本地 → 容器）
cd $REPO_ROOT/release/ascend/port910b/attempts
docker cp X_probe_forms.py  cann910b-k:/tmp/ && docker cp X_probe_nd2nz.py cann910b-k:/tmp/ && docker cp X_probe_slots.py cann910b-k:/tmp/
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; cd /tmp && python3 /tmp/X_probe_forms.py'
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; cd /tmp && python3 /tmp/X_probe_nd2nz.py'
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; cd /tmp && python3 /tmp/X_probe_slots.py'
docker cp X_carve_hivmc.py cann910b-k:/tmp/ && docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; python3 /tmp/X_carve_hivmc.py'   # 产物 /tmp/xcarve/dump0.ll

# 统一编译命令（所有探针共用；bisheng 需先 source set_env.sh，否则报 "can not find ASCEND_HOME_PATH"）
bisheng -std=c++20 -fPIC -O2 --npu-arch=dav-2201 \
  -I/tilelang/src -DTL_PORT910B_NATIVE_TYPES \
  -I/usr/local/Ascend/cann-8.5.0/aarch64-linux/include \
  -I/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl \
  -I/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/include \
  --cce-aicore-only -c <tu>.asc -o <tu>.o
```

探针 TU 形态约束（踩过的坑，写给后来者）：
- 必须是 `extern "C" __global__ __cube__ void probe(__gm__ T *g)`；`__kernel__` 未知类型。
- `__cbuf__` 指针**不能**作 `__global__` 形参（"not allowed to have CCE memory attribute"）→ 与 §12 同款用 `(uintptr_t)0x10000` 中转。
- `-S` 在 aicore 子进程不支持（会产出 host-only `.s`，极易误判 PASS）→ 用 `--cce-aicore-only -c`。
- 取 rc 时**不要让管道吞掉退出码**（`| head` 会让 `echo rc=$?` 变成管道尾的 rc）→ stderr 落文件后再判。本项目 P1-1d 的"三形全 PASS"假象正是这个坑，纠正后真相为 **B1/B2/B3 全 rc=1、仅 C1 rc=0**。

---

## 2. 路径 3（bisheng sema 判别）——2201 真实签名

### 2.1 3101 头在 2201 不可见（头文件级门）

```
/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_types.h:231
    #if (__NPU_ARCH__ == 3101)
    #include "cce_aicore_intrinsics_3101.h"
```
2201 侧的 intrinsic 声明全在 `cce_aicore_intrinsics.h`，且**均为变长 alias**（故签名校验发生在 builtin 的 sema 里，不在 C++ 重载里 —— 这正是"报错文本=签名判别器"成立的原因）：
```
cce_aicore_intrinsics.h:1384  __attribute__((clang_builtin_alias(__builtin_cce_load_cbuf_to_ca))) void load_cbuf_to_ca(...);
cce_aicore_intrinsics.h:974   ... void copy_gm_to_cbuf(...);
cce_aicore_intrinsics.h:984   ... void copy_gm_to_cbuf_multi_nd2nz_b16(...);   ← 2201 有带后缀的 ND2NZ 件
（同族：_b8 / _b32s / _multi_dn2nz / copy_gm_to_cbuf_v2 / load_gm_to_cbuf_2dv2 等，见 :976-990,:1436-1446）
```
3101 显式形与 2DV2 打包原文（**任务书引用的行号需修正**：`_3101.h:4233-4263` 是 `set_l1_2d` 族〔L 代理已定〕，`load_cbuf_to_ca` 显式形在 :3018-3042）：
```
cce_aicore_intrinsics_3101.h:3018  // ASM: LOAD_L1_TO_L0A_2DV2.b16 [dst], [src], config0, config1, #transpose
cce_aicore_intrinsics_3101.h:3022  void load_cbuf_to_ca(__ca__ bfloat16_t*, __cbuf__ bfloat16_t*, uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride, bool transpose);
cce_aicore_intrinsics_3101.h:3023  // -> cfg0=(mStart&0xffff)<<0 | (kStart&0xffff)<<16 | (mStep&0xff)<<32 | (kStep&0xff)<<40, cfg1=(srcStride&0xffff)<<0 | (dstStride&0xffff)<<16
```

### 2.2 三种候选形在 2201 的 sema 结果（互斥错误即签名）

| 探针 | 发的形 | rc | 报错原文（关键行） |
|---|---|---|---|
| `B1_ca_p11d_form` | V1 9 参，但 sid 槽给 105 | **1** | `error: the range of 7th parameter must be [0, 15]` |
| `B5_ca_p11d_sidinrange` | V1 9 参，sid=2、transpose=`true` | **0** | —（编过，产 `/tmp/dev_B5.o` 2888B） |
| `B2_ca_explicit_form` / `X1_explicit_on_2201` | 3101 显式 9 参 `(u16,u16,u8,u8,i16,u16,bool)` | **1** | `error: the 9th parameter must be a type '__cce_scalar::addr_cal_mode_t'` |
| `B3_ca_config_form` | `(dst,src,u64 cfg0,u64 cfg1,bool)` | **1** | `error: the 5th parameter must be a type '__cce_scalar::addr_cal_mode_t'` |
| `B4_cb_explicit_form` | cb 侧 3101 显式形 | **1** | 同族错误 |

判读：B2/B3 的报错都要求**多出一个 `addr_cal_mode_t` 尾槽**，即 2201 上的 checker 是按 **V1 族的位序**在数槽；3101 的 2DV2/显式形在 2201 没有对应重载。

### 2.3 逐槽类型/取值区间（E 系列，`X_probe_slots.py`）

基准 E0 = V1 合法 9 参，`rc=0`；然后逐槽放 `1.5f` 逼出期望类型：

```
E1_slot3_float   FAIL  error: the 3rd  parameter maybe need a type 'unsigned short'   ← startIndex
E2_slot4_float   FAIL  error: the 4th  parameter maybe need a type 'unsigned char'    ← repeatTimes（8-bit！）
E3_slot5_float   FAIL  error: the 5th  parameter maybe need a type 'unsigned short'   ← srcStride
E4_slot6_float   FAIL  error: the 6th  parameter maybe need a type 'unsigned short'   ← dstGap
E5_slot7_..16    FAIL  error: the range of 7th parameter must be [0, 15]               ← sid（4-bit）
E6_slot8_int     PASS （transpose 槽接受常量整数字面量）
E7_arity_8args   FAIL  error: function type 'void (__ca__ __bf16 *, __cbuf__ __bf16 *, unsigned short,
                         unsigned char, unsigned short, unsigned short, bool, unsigned int) noexcept'
                         of 'load_cbuf_to_ca' does not support the given target feature  ← 少一个 addrMode 就落到"不支持"
E8_arity_10args  PASS （第 10 参 inc 可给；与官方 Cal 的 10 参调用一致）
```
→ 与 `dav_c220/kernel_operator_mm_impl.h:24-37` 的 V1 Cal 完全对齐：
```cpp
LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& p) {
    load_cbuf_to_ca(dst, src, p.startIndex, p.repeatTimes, p.srcStride, p.dstGap, p.sid, /*transpose*/0或1, inc);
}
```
且 `LoadData2DParams` 字段类型（`include/basic_api/kernel_struct_mm.h`）与之逐槽吻合：
```cpp
struct LoadData2DParams { uint16_t startIndex=0; uint16_t dstGap; uint16_t srcStride;
                          bool ifTranspose; uint8_t repeatTimes; uint8_t sid; uint8_t addrMode; };
// 构造序：(startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrMode)
```

### 2.4 运行期值**不受**区间检查（507015 的静默通道）

```
/tmp/xsl_rt3.asc: load_cbuf_to_ca(d,c,(u16)g[0],(u8)g[1],(u16)g[2],(u16)g[3],(uint8_t)sid,false,(addr_cal_mode_t)0)
                  其中 sid = g[4]（运行期 int32，可 >15）
→ rc=0（无任何告警）
/tmp/xsl_rt4.asc: 仅把 transpose 换成运行期 (bool)g[5]
→ rc=1，报错回打了 3101 形整张类型表：
   'void (__ca__ __bf16 *, __cbuf__ __bf16 *, unsigned short, unsigned short, unsigned char, unsigned char, short, unsigned short, bool) noexcept' ... does not support the given target feature
```
判读（两条都用得上）：
1. **sid 越界是纯运行期故障**，编译期零保护 → 只要 §12 的 sid 槽收到 `m_start` 且 `m_start>15`，就是非法 L1 分区号，正好是 DMA 非法访问类异常。
2. transpose 必须是编译期常量（§12 现在恒传字面量 `false`，与该约束自洽；codegen 也是靠第 9 个 IR 参在**选件名**而非传值）。

### 2.5 GM→L1 各件在 2201 的可编性（D 系列，`X_probe_nd2nz.py`）

```
D1_nd2nz_b16_official11   PASS   copy_gm_to_cbuf_multi_nd2nz_b16(dst,src,sid, ndNum,nValue,dValue,
                                     srcNdMatrixStride,srcDValue, dstNzC0Stride,dstNzNStride,dstNzMatrixStride)
D5_nd2nz_b32s             PASS   （float 变体同样可用）
D6_nd2nz_b8               PASS   （int8 变体）
D7_gm2l1_sec12_8arg       PASS   copy_gm_to_cbuf(dst,src,(int8_t)0,(u16)nRows,blk,blk,blk,(pad_t)0) —— §12 现用形
D2_nd2nz_b16_sidOOB       FAIL   error: the range of 3rd parameter must be [0, 15]        ← ND2NZ 的 sid 也在第 3 槽、同样 4-bit
D3_nd2nz_b16_slot5_float  FAIL   error: the 5th parameter maybe need a type 'unsigned short' ← 全 uint16_t 口径
D4_multi_nd2nz_3101_9arg  FAIL   error: function type 'void (__cbuf__ __bf16 *, __gm__ __bf16 *, unsigned char,
                                     unsigned long, unsigned char, unsigned short, unsigned int, unsigned long, bool)
                                     noexcept' of 'copy_gm_to_cbuf_multi_nd2nz' does not support the given target feature
```
→ **3101 的 9 参 `copy_gm_to_cbuf_multi_nd2nz`（§12 件名 `asc_copy_gm2l1_nd2nz` 的原型暗示）在 2201 明确"不支持该 target feature"**；2201 要 ND→NZ 必须走带位宽后缀的 **11 参**件。
→ `copy_gm_to_cbuf` 8 参裸形在 2201 **能编**，所以它不是编译期病灶；问题在语义（§6）。
→ `load_gm_to_cbuf_2dv2`（C3 探针）在本树只被 `dav_m310/kernel_operator_mm_impl.h:253/263` 使用，**dav_c220 不使用**。

---

## 3. 路径 1（hivmc 内嵌 bitcode 反解）——降级签名的旁证

方法同 L 代理（`X_carve_hivmc.py`）：`hivmc` 里按 `BC\xc0\xde` 切 5 块 → `llvm-link` → `bisheng -x ir -S -emit-llvm` → 读 `!asan.cce.api.name` 与 `__sanitizer_report_*` 的 mangled 名。chunk0(1520168B) 命中，产物 `/tmp/xcarve/dump0.ll`(10.7MB)。

复跑取证据行：
```bash
docker exec cann910b-k grep -n "sanitizer_report_load_cbuf_to_ca\|sanitizer_report_copy_gm_to_cbuf" /tmp/xcarve/dump0.ll
```
实测（本窗原文）：
```
!141 = !{!"_Z50__sanitizer_report_copy_gm_to_cbuf_multi_nd2nz_b16PU3AS1hmmlPU3AS2DhPU3AS1Dhmm"}
!146 = !{!"_Z34__sanitizer_report_load_cbuf_to_caPU3AS1hmmlPU3AS3DhPU3AS2Dhmbj"}
!423 = !{!"_Z34__sanitizer_report_copy_gm_to_cbufPU3AS1hmmlPU3AS2vPU3AS1vmj"}
!436 = !{!"_Z34__sanitizer_report_load_cbuf_to_caPU3AS1hmmlPU3AS3aPU3AS2ambj"}
!510 = !{!"_Z44__sanitizer_report_load_cbuf_to_ca_transposePU3AS1hmmlPU3AS3aPU3AS2amm"}
```
解码（AS1=GM，AS2=CA，AS3=CBUF/L1，AS4=CB；`m`=u64，`j`=u32，`b`=bool，`Dh`=bf16，`a`=int8；公共前缀 `PU3AS1h m m l` 是 sanitizer 自身头部）：

| 件 | 尾随实参（去掉公共前缀） | 读法 |
|---|---|---|
| `load_cbuf_to_ca` bf16 | `(src AS3 bf16*, dst AS2 bf16*, **m**, b, j)` = (cfg0 u64, transpose bool, addrMode u32) | **单条 cfg 打包** ⇒ V1 族（startIndex/repeatTimes/srcStride/dstGap/sid 打进一个 64-bit 描述符）|
| `load_cbuf_to_ca` int8 | 同上（`AS3a/AS2a`）| dtype 只换指针元素类型，cfg 结构不变 |
| `load_cbuf_to_ca_transpose` | `(..., m, m)` = **两条 u64** | 转置件是双字 cfg 打包（与 §12 现按 8 参传值的位序需另行核对，见 §8-E5）|
| `copy_gm_to_cbuf` | `(dst AS2 void*, src AS1 void*, m, j)` | 8 参裸形被降为 (cfg0, pad)，块式描述符 |
| `copy_gm_to_cbuf_multi_nd2nz_b16` bf16 | `(dst AS2 bf16*, src AS1 bf16*, m, m)` | 11 参 ND2NZ 降为 **两条 cfg u64** ⇒ 与 `_3101.h:1337` 的 `cfg0/cfg1` 位段注释同构 |

判读：编译器内部对 2201 的 L1→L0A **只降出一个 config 字**，与 3101 的 `LOAD_L1_TO_L0A_2DV2`（**两个** config 字 cfg0/cfg1）形态不同 —— 独立旁证 §2.2 的"2201 不吃 2DV2/显式形"。
（注：hivmc 侧只能拿到降级后的类型，拿不到 asm 位表；位段的权威来源仍是 `_3101.h` 注释〔3101〕与 sema/官方 Cal〔2201〕。此差异已在 §8 标为未做项 E6。）

---

## 4. 跨 arch 交叉验证的**环境受限**说明

```
bisheng --npu-arch=dav-3101  → 接受该字符串（非 "Unsupported NPU architecture"），但**空 TU 也编不过**：
    /tmp/xsl_min.asc:3:107: error: expected ')'      ← 报错位置在头文件展开内，源文件仅 3 行
bisheng --npu-arch=dav-5102 / dav-c220 → error: Unsupported NPU architecture or soc
```
即：**本容器（cann910b-k，CANN 8.5.0 aarch64 for 910B）不具备 dav-3101 的可用头集**，无法在 3101 上正向编出显式形作为对照。
→ 因此"3101 形只属 3101"的结论靠三件独立证据支撑，不依赖该对照：
(a) `__clang_cce_types.h:231` 的 `#if (__NPU_ARCH__ == 3101)` 包含门；
(b) A1/B2/B3 的 2201 sema 结果（§2.1/2.2）；
(c) hivmc 降级形态差异（§3：单 cfg vs 双 cfg）。

---

## 5. 官方实现树取证（替代路径 2 的更强证据）

### 5.1 V1 槽语义的**官方文档原文**（单位裁决）
`asc/include/interface/kernel_operator_mm_intf.h:26-36`
```
 * @param [in] loadDataParams.startIndex  Fractal matrix ID            ← 分形矩阵编号，不是元素偏移
 * @param [in] loadDataParams.repeatTimes repeat times
 * @param [in] loadDataParams.srcStride   src block stride             ← 块步距（分形块单位）
 * @param [in] loadDataParams.sid         SMMU SID
 * @param [in] loadDataParams.dstGap      interval between the previous tail and the next fractal head
 * @param [in] loadDataParams.ifTranspose enable parameters of transpose function
```
`asc/include/interface/kernel_operator_data_copy_intf.h:44-56`（GM→L1 ND2NZ 字段单位原文）
```
 * @param [in] intriParams.ndNum            nd number of data to be moved
 * @param [in] intriParams.nValue           n value
 * @param [in] intriParams.dValue           d value in unit of element
 * @param [in] intriParams.srcNdMatrixStride stride between nd matrixs at source ND matrix in unit of element
 * @param [in] intriParams.srcDValue        SRC_D value in unit of element     ← GM 行宽（元素）
 * @param [in] intriParams.dstNzC0Stride    stride of nz between 2 C0 in L1 in unit of C0_size
 * @param [in] intriParams.dstNzNStride     stride of n between 2 C0 in L1
 * @param [in] intriParams.dstNzMatrixStride DST_nz_matrix_stride in L1 in unit of element
```

**地址模型的硬证据**（单位不再靠猜）：`asc/impl/basic_api/utils/kernel_check_data_copy_overflow.h:505-520`
```cpp
uint64_t srcMaxOffsetBytes = dataCopyParams.blockCount * AlignUp(dataCopyParams.blockLen, DEFAULT_C0_SIZE)
                           + (dataCopyParams.blockCount - 1) * dataCopyParams.srcStride * DEFAULT_C0_SIZE;
uint64_t dstMaxOffsetBytes = (nd2nzParams.ndNum - 1) * nd2nzParams.dstNzMatrixStride * sizeof(PrimT<T>)
                           + (nd2nzParams.nValue - 1) * nd2nzParams.dstNzNStride * DEFAULT_C0_SIZE
                           + (DivCeil(nd2nzParams.dValue*sizeof(PrimT<T>), DEFAULT_C0_SIZE) - 1)
                               * nd2nzParams.dstNzC0Stride * DEFAULT_C0_SIZE + DEFAULT_C0_SIZE;
```
`asc/impl/basic_api/utils/kernel_utils_constants.h:32`: `const int32_t DEFAULT_C0_SIZE = 32;`
`asc/include/adv_api/transpose/confusion_transpose_tiling.h:26/31`: `BLOCK_CUBE = 16`、`ONE_BLK_SIZE = 32`
→ **`copy_gm_to_cbuf` 的 blockLen/srcStride/dstStride 单位 = 32B 块**；**`dstNzC0Stride`/`dstNzNStride` 单位 = 32B 块**（`dstNzMatrixStride` 单位 = 元素）。这直接判掉了任务书 V2 假设里"元素"的一侧：**GM/L1 侧一切 stride/len 都是 32B 块口径**。（V1 `load_cbuf_to_ca` 的 srcStride 则是**分形块数**口径，见 §5.2。）

### 5.2 官方 V1 正确调用模板（L1→L0A，非转置）
`asc/impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_load2d.h:41-110`
```cpp
uint16_t blockUseM = Ceil(madM, BLOCK_CUBE);      // M 向分形块数
uint16_t blockUseK = Ceil(madK, c0Size_);         // K 向 C0 组数
srcL1Offset = c0Size_ * aL1MOffset + aL1M * aL1KOffset;      // ← 起点折进指针（不进任何槽！）
LoadData2dParams loadDataParams;
loadDataParams.repeatTimes = blockUseK;                       // ← repeat 沿 K(C0) 方向
loadDataParams.srcStride   = Ceil(aL1M, BLOCK_CUBE);          // ← L1 的 row16 块数 = C0 组间距
loadDataParams.ifTranspose = false;
if (blockUseK == 1) { repeatTimes = blockUseM; srcStride = 1; LoadData(dst, aMatrix[srcL1Offset], p); }
else for (int i = 0; i < blockUseM; i++)                       // ← M 向靠外层 for 拆趟
        LoadData(dst[i*dstOffset], aMatrix[srcL1Offset + i*srcOffset], p);   // dstOffset = blockUseK*CUBE_MAX_SIZE/factor_
```
转置 A 与 L0B 同构（`load_to_l0b/load_to_l0b_load2d.h:38-92`、`load_to_l0b_basic.h:111-119`）：
```cpp
// load_to_l0b_basic.h:112  注释即构造序
// startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
LoadData2dParams loadDataParams{0, l0bRepeat, l0bSrcstride, 0, l0bDststride, 0, 0};
for (uint64_t i = 0; i < l0bLoop; i++) { LoadData(l0B[l0bOffset], l1B[l1bOffset], p); l1bOffset += ...; l0bOffset += ...; }
```
**四条硬规则**（§12 三条违反）：
1. `startIndex` 恒 0，**二维起点折进 src 指针**；
2. `repeatTimes` 只沿 **K/C0** 方向（非转置 A、B），M/N 方向用**外层 for 拆趟**；
3. `srcStride` = L1 的 **row16 块数**（= C0 组间距，分形块单位）；
4. `sid` 官方恒 0（未做 L1 分区）→ 任何非 0 都是"自选分区"，必须真机验。

### 5.3 V2 在 2201 是"运行期 NOT_SUPPORT"，不是编译期缺件
```
asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:94/100/106
  LoadData2DL12L0ACal(..., const LoadData2DParamsV2&) → ASCENDC_REPORT_NOT_SUPPORT(false,"LoadData with LoadData2DParamsV2 from A1 to A2")
  L12L0BCal → "... from B1 to B2"；GM2L0ACal → "... from GM to A2"
asc/impl/basic_api/kernel_operator_mm_base_impl.h:~229  5102-only：LoadData(dst, GlobalTensor, LoadData2DParamsV2, Nd2NzParamsV2)
asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_nz.h:49-73  CopyNZ2NZDecompMode 全体包在 #if __NPU_ARCH__ == 5102
```
→ tilelang IR 的 V2 语义（mStart/kStart/mStep/kStep/srcStride/dstStride）在 910B **只能靠"折指针 + 拆趟循环"手工降级**，不存在一条等价硬件指令。这也解释了 P1-1d 为什么"补到 9 参仍崩"。

### 5.4 GM→L1 的官方两路（§6 裁决的主证）
`asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h:83-108`（ND 线性块拷贝）
```cpp
DataCopyGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& p) {
    ASCENDC_CHECK_TENSOR_PTR_ALIGN(dst, TPosition::A1, ONE_BLK_SIZE, ...);
    copy_gm_to_cbuf((__cbuf__ void*)dst, (__gm__ void*)src, (int8_t)0,
        (uint16_t)p.blockCount, (uint16_t)p.blockLen, (uint16_t)p.srcStride, (uint16_t)p.dstStride, (pad_t)0);
}
```
`同文件 :263-295`（ND→NZ 硬件转换）
```cpp
DataCopyGM2L1ND2NZImplBase(...)  // b16：
  copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ T*)dst, (__gm__ T*)src, 0, p.ndNum, p.nValue, p.dValue,
      p.srcNdMatrixStride, p.srcDValue, p.dstNzC0Stride, p.dstNzNStride, p.dstNzMatrixStride);
```
官方 Matmul 侧（L1 目的地）全部走 NZ：`.../copy_cube_in/copy_tile_to_cube/data_copy_wrapper_nd.h:41-115`（ND 输入 → `CopyND2NZ`）与 `data_copy_wrapper_nz.h:39-80`（NZ 输入 → `CopyNZ2NZ`）。2201 还有专属越界兜底：
```cpp
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    if (gCol >= UINT16_MAX) { nd2nzParams.nValue = 1; nd2nzParams.srcDValue = width;
        for (int32_t i = 0; i < height; ++i) DataCopy(dst[i*c0Size_], src[srcOffset + gCol*i], nd2nzParams); }
    else DataCopy(dst, src[srcOffset], nd2nzParams);
```
`data_copy_wrapper_nd.h:93` 给出 dstNzC0Stride 的标准取值：`dstNzC0Stride = Ceil(height, BLOCK_CUBE) * BLOCK_CUBE;`（`dstNzNStride = 1`、`ndNum = 1`、`srcNdMatrixStride = 0`）。

---

## 6. 裁决②：GM→L1 是否需 NZ —— **需要**

证据合拢：
1. V1 `load_cbuf_to_ca.startIndex` 文档语义 = **Fractal matrix ID**（§5.1）→ 消费侧按分形块寻址。
2. 官方 Matmul 写 L1 **只有** ND2NZ / NZ2NZ 两路（§5.4），无任何"ND 直写 L1 喂 cube"的路径。
3. 2201 存在并被官方 dav_c220 实现使用的 ND→NZ 件 = `copy_gm_to_cbuf_multi_nd2nz_b16/_b8/_b32s`，本窗实证可编（D1/D5/D6）。
4. §12 `asc_copy_gm2l1_nd2nz` 的调用与官方 `DataCopyGM2L1Impl`（ND 线性块拷贝，非格式转换）**逐字同款**，且 `(void)cols` 把 IR 的 `d_value` 丢掉、`asc_set_gm2l1_nz_para` 是**空实现**（`port910b_compat.h:860-862`）→ NZ 的 c0 pitch 没有任何地方生成。

→ **V1 假设成立**：950 件名 `asc_copy_gm2l1_nd2nz` 所暗示的"ND→NZ"在 910B 上并未发生；L1 拿到的是行主序 ND，而 L1→L0 按分形块寻址去读 → 地址/长度双错。
→ V2 假设（stride 单位元素 vs 32B 块）：§5.1 的地址模型已判定 **GM/L1 侧=32B 块、L0 侧 srcStride=分形块数**；§12 现用 `blk = rowBytes>>5` 在"块"这一侧是对的（所以它编得过、也搬得动 ND 行），错的是**布局与槽位**，不是 blk 换算本身。

---

## 7. 裁决③：§12 逐槽对照表

### 7.1 发射链（真源）
`src/ascend/op/copy.cc:457-464,535-538`（L1→L0A/B）
```cpp
m_start = src_region.outer1->min;  k_start = src_region.outer0->min;      // 分形块索引（row16 / C0 组）
m_step  = src_region.outer1->extent; k_step = src_region.outer0->extent;  // 分形块数 / C0 组数
src_stride = src_info.outer1;        // L1 全缓冲 row16 外层 extent（= C0 组间距，分形块数）
dst_stride = dst_region.outer1->extent;
ld_args = {dst_ptr, src_base_ptr, m_start, k_start, m_step, k_step, src_stride, dst_stride, I(needs_transpose)};
```
`src/ascend/layout/ascend_layouts.cc:362-366,370-381`：物理尾 4 轴 = **(outer0=C0 组, outer1=row16, row_inner=16, col_inner=C0)** → 上述量**已是分形块口径**（这点很重要：单位本来就对）。
`src/ascend/codegen/codegen_ascend.cc:1568-1575`：`asc_copy_l12l0a` 收到 args[0..7]（**args[8] transpose 只用来选件名**）。

### 7.2 错位表（`port910b_compat.h:891-898`，`900-906`）

| 硬件槽（§2.3 实证） | 宽度/约束 | wrapper 收到的值 | 语义判定 |
|---|---|---|---|
| 1 `dst` | `__ca__ T*` | `dst` | ✔ |
| 2 `src` | `__cbuf__ T*` | `src`（IR `src_base_ptr`）| ⚠ **起点未折进来**（官方规则 1）|
| 3 `startIndex` | u16 | `(uint16_t)kStart` ← IR `k_start` | ✘ 轴向/用法错：官方恒 0；二维起点无法用单一线性 ID 表达 |
| 4 `repeatTimes` | **u8**（≤255） | `(uint8_t)mStep` ← IR `m_step` | ✘ **轴选反**：应 = `k_step`（C0 组数）；M 向靠循环。IR `m_step`>255 还会 8-bit 截断 |
| 5 `srcStride` | u16 | `(uint16_t)srcStride` ← IR `src_stride` | ✔ 口径与官方 `Ceil(aL1M,16)` 同为"row16 块数"，但与 repeat 的轴向不配套 |
| 6 `dstGap` | u16 | `(uint16_t)dstStride` ← IR `dst_stride` | ？官方 dstGap="上一趟尾→下一分形头间隔"，与 L0 row16 extent 不等价；官方靠 `dst[i*dstOffset]` 循环 → 待 §8-E4 标定 |
| 7 **`sid`** | **4-bit，常量才检查 [0,15]** | `(uint8_t)sid` ← **codegen args[2] = IR `m_start`** | ✘✘ **致命**：`m_start` 是 row16 块起点（0..127 量级），运行期无区间保护（§2.4 RT3 rc=0）→ 非法 L1 分区号 = **507015 头号嫌疑** |
| 8 `transpose` | 必须编译期常量 | `false` 字面量 | ✔（与件名选择机制自洽）|
| 9 `addrCalMode` | `addr_cal_mode_t` | `(addr_cal_mode_t)0` | ✔ |
| 10 `inc` | 可省（E8 证）| 官方传全局 `inc`，§12 未传 | ✔（等价）|
| — | — | **`(void)kStep`** | ✘ 被丢弃的正是官方唯一合法的 `repeatTimes` 值 |
| — | — | **无 M 向拆趟循环** | ✘ 单条指令无法覆盖 m_step>1 的行块（官方规则 2）|

wrapper 形参名与 IR 语义的**命名错位**是根因放大器：`asc_copy_l12l0a(dst, src, int sid, int kStart, ...)` 的第 3 形参叫 `sid`，而 codegen 传的是 `m_start`；第 4 形参叫 `kStart`，传的是 `k_start` —— 名字看着"对"，实际整条槽表在两套语义间平移。

### 7.3 GM→L1 槽表（`port910b_compat.h:868-876`）

| 现发（`copy_gm_to_cbuf` 8 参） | 值 | 判定 |
|---|---|---|
| sid | `(int8_t)0` | ✔ |
| blockCount | `(uint16_t)nRows` ← IR `n_value`(=源行数) | 行块数，口径对（32B 块数 vs 块数需与 blockLen 同单位口径核对）|
| blockLen | `blk = rowBytes>>5`（=16 元素 bf16 行）| ✔ 单位（32B 块，§5.1）|
| srcStride / dstStride | 都 = `blk` | = 连续 ND 行，**产出 ND** |
| pad | `(pad_t)0` | ✔ |
| IR `d_value`(cols) | **`(void)cols`** | ✘ NZ 的 d 维信息被丢 |
| IR `nz_c0_stride`(args[10]) | 只喂给**空实现** `asc_set_gm2l1_nz_para` | ✘ c0 pitch 从未落地 |

---

## 8. 未定论项 + 剩余最小判别实验设计（供下窗一次做完）

**已排除**：① 位序（P1-1d 已按官方 V1 补齐，本窗证 V1 是唯一可编形）；② 3101 显式/config 形误用（2201 target-feature 直接拒）；③ `blk=rowBytes>>5` 的 32B 单位口径（官方地址模型坐实）；④ GM→L1 件的可编性（8 参与 11 参都能编，故非编译病灶）。

**仍未定论（本窗无法在只读+编译面判定）**：
- U1 §12 单条指令 → "折指针 + 拆趟"后，L1 源指针偏移的**块线性序**（`block_id = k_block*pitch + m_block`，据 `srcStride=Ceil(aL1M,16)` 沿 K 前进反推；与 tilelang phys 轴序一致，但**未上卡验**）。
- U2 L0 目的侧步进 `dstOffset = blockUseK*CUBE_MAX_SIZE/factor_`（`CUBE_MAX_SIZE` 定义未在本次抓取中命中）→ 需要字节口径标定。
- U3 `dstGap` 是否应恒 0（官方模板里 `dstGap` 在 L0A 路为 0，在 `load_to_l0b_basic` 里 = `nFraC0-1`）。
- U4 ND2NZ 与 `asc_fill_l1`（§12 补件，32B 块口径）在 pad 区上的配合。
- U5 `asc_copy_l12l0a_transpose` 的 8 参位序（§3 显示 transpose 件是**双 cfg 字**，族属与 `_3101.h:3402` 同，2201 位序未取到）。
- U6 hivmc 侧只拿到降级类型，未做 asm 位表（2201 V1 config 字的**精确位段**仍靠官方 Cal 的字段序推，未逐位实证）。

**最小判别实验（按信息量/成本排序，建议一次真机窗跑完）**

| 实验 | 目的 | 做法（都在 attempts/ 新 TU，不触产品） | 判据 |
|---|---|---|---|
| **E-3（首选，单条改动）** | 隔离 sid 是否首要根因 | §12 现形不动，仅把第 7 参强制 `(uint8_t)0`（其余原样）编一版对照件 | 崩溃消失 → sid 越界坐实为首要根因；仍崩 → 布局（ND vs NZ）为必要前提。**一次真机即可二选一** |
| **E-1** | 证 GM→L1 产出是否 NZ | 16×16 bf16 单块 kernel：GM(ND) →`copy_gm_to_cbuf_multi_nd2nz_b16`→ L1 → `copy_cbuf_to_gm` 回 GM，与手工 NZ 期望位型逐字节比 | 位型 == 16×16 分形块 ⇒ ND2NZ 件与参数表正确 |
| **E-2** | 提案件端到端 | 按 `X_sec12_proposal.md` 的折指针+拆趟件跑 16×16×16 mad 对拍 | rel < 1e-3 ⇒ U1/U2/U3 全部被同一实验收敛 |
| **E-4** | 标定 L0 步进/`dstGap` | 同 E-2，但只发 1 趟（k_step=1、m_step=1），dump L0A 前 1KB | 得 `CUBE_MAX_SIZE` 口径与 dstGap 真值 |
| **E-5** | transpose 件位序 | 单发 `load_cbuf_to_ca_transpose` 的 8 参/9 参两版，编译面看 sema + 真机 dump L0A | 定 §8-U5 |
| **E-6** | V1 config 位段逐位实证 | 在容器内对 `xsl_E*_*.o` 用 `llvm-objdump -d --section=.text`（hiipu 反汇编当前 `<not available>`）失败 → 改走 hivmc 反解 `_mlir_ciface_load_cbuf_to_ca` **实现体**（本次只取了名字表，未取函数体）| 拿到 2201 打包位序 |

> 说明：E-1/E-2/E-4 需要真机卡时；**E-6 可在本容器离线完成**（下一步若给离线窗，优先做 E-6 + 把 `CUBE_MAX_SIZE`/`c0Size_` 常量抓全）。

---

## 9. 证据索引（文件:行 → 本窗命令）

| 事实 | 位置 | 复跑 |
|---|---|---|
| 2201 只吃 V1 9 参 | `dav_c220/kernel_operator_mm_impl.h:24-37,94-108` | `docker exec cann910b-k sed -n '18,70p' .../dav_c220/kernel_operator_mm_impl.h` |
| V1 槽类型/区间 | sema 实测 E1-E8 | `python3 /tmp/X_probe_slots.py` |
| 运行期 sid 无保护 | `/tmp/xsl_rt3.asc` rc=0 | 见 §1 编译命令 |
| startIndex=Fractal matrix ID | `include/interface/kernel_operator_mm_intf.h:26-36` | `sed -n '18,40p'` |
| V1 拆趟模板 | `.../load_to_l0a/load_to_l0a_load2d.h:41-110`；`load_to_l0b_basic.h:112` | `sed -n '30,110p'` |
| 32B 块单位 | `kernel_check_data_copy_overflow.h:505-520`；`kernel_utils_constants.h:32`；`confusion_transpose_tiling.h:26/31` | §5.1 |
| GM→L1 两路 | `dav_c220/kernel_operator_data_copy_impl.h:83-108, 263-295` | §5.4 |
| ND2NZ tiling 约定 | `.../copy_tile_to_cube/data_copy_wrapper_nd.h:41-115`（含 `#if __NPU_ARCH__==2201` 兜底）| §5.4 |
| 2201 有 11 参 ND2NZ 件 | D1/D5/D6 rc=0；D4 rc=1 | `python3 /tmp/X_probe_nd2nz.py` |
| 3101 形位段/门 | `cce_aicore_intrinsics_3101.h:3018-3042`；`__clang_cce_types.h:231` | §2.1 |
| hivmc 降级签名 | `/tmp/xcarve/dump0.ll` `!141/!146/!423/!436/!510` | §3 |
| tilelang IR 槽序 | `src/ascend/op/copy.cc:457-464,535-538`；`codegen_ascend.cc:1568-1575`；`ascend_layouts.cc:362-381` | 本地 |
| §12 现形 | `src/tl_templates/ascend/port910b_compat.h:860-876, 891-906, 1049-1057` | 本地 |
