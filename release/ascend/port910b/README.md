# README — 910B 兼容层的接入方式（p2-13 / P0-1 产物）

配套判决文档：`INVENTORY.md`（950-only 依赖面穷举，349 个去重符号）
逐符号附表：`inventory_symbols.md`（由 `./checklist.sh symbols` 生成，勿手改）

---

## 1. 本目录文件与角色

| 文件 | 角色 | 本机能否执行 |
|---|---|---|
| `INVENTORY.md` | 判决正文：A/B/C/D 四分类 + 逐组处置方案 + 五道墙 + 七算子矩阵 | 只读 |
| `inventory_symbols.md` | 逐符号附表（举证 `文件:行` + 命中数 + 出现文件） | 只读，自动生成 |
| `checklist.sh` | **自证脚本**：族穷举 → 反查清单（audit）、附表生成（symbols）、族外残差体检（gap） | ✅ 可跑（纯 grep） |
| `port910b_compat.h` | B 类 20 符号的兼容层实现草稿（bf16/fp16 类型桥 + 转换内建 + 锁模式常量 + 空宏） | 主机 `-fsyntax-only` ✅ |
| `host_selfcheck.cc` | 兼容层的主机侧等价性自检（位宽/RNE 数值/lane 位视图），**非 ascend 目标** | ✅ 可编译可跑 |
| `probe910b.sh` | 上卡探针 P1–P10（裁决 44 个 D 类符号与墙0/墙3/墙4） | ❌ 本机拒绝执行（exit 2） |

### 目录结构（2026-10-12 整理收纳）

| 位置 | 内容 | 说明 |
|---|---|---|
| 根 | 在用件：`card_run.sh`（卡侧引导主件）、`run_delta_card.py` / `run_numerics.py` / `gradcheck` 相关判决件、`launch_card_local.sh` / `wait_card_local.sh` / `fixup_card.sh` / `bundle_ondemand.sh`（发射辅助）、上表 6 件 | 窗九及后续发射直接用 |
| `windows/` | 窗二至窗八历史判决发射器：`windowN_card.sh`（卡侧）× `run_windowN_local.sh`（宿主侧）× `poll_*.sh`（轮询）共 15 件 | 各窗凭据与判决行见 `release/runs/1011-*/notes.md`；脚本内硬编码绝对路径指向整理前位置，**仅作历史归档不再发射**，复现以 git 历史为准 |
| `archive/` | 一次性诊断件：`ssh_probe.sh` / `ssh_diag.sh` / `ssh_retry.sh`（网关 502 三件）、`res_check_local.sh`（实例资源盘点） | 经验结论已录入各窗 notes |
| `attempts/`、`patches/` | 早期探针结构与补丁 | 未动 |
| `cann_cache/`（gitignored） | CANN 8.5.0 toolkit run 包 | 容器 `cann910b-l` 的安装源；**卡上为 8.5.2，容器为 8.5.0——此版本差是窗九"卡上编译差异"分支的已知输入** |


---

## 2. 兼容层放哪（两条路，推荐 A）

### 路 A：**零改主仓**，用 `-include` 注入（P0-2 开局用这条）

主仓 `src/` 是 `-I` 根（`tilelang/env.py:650` 赋值 `TILELANG_TEMPLATE_PATH`，声明处 `:367`），把本目录也变成一个
include 根，然后用 `-include` 让兼容头在每个 device TU 最前面生效：

```python
# 用户侧（kernel 构造或 jit 配置），三行搞定：
pass_configs = {
    PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
        "-I/path/to/system-one/release/ascend/port910b",   # 让头可被找到
        "-include", "port910b_compat.h",                   # 必须先于模板头
        "-DTL_PORT910B_BF16_BUILTIN=0",                    # 见 §3，按 P9 结果定
    ],
}
# 走 tilelang/ascend/backend.py:18（normalize_options 读的就是这个 key）
# 与 tilelang/jit/adapter/libgen.py:154 同一条通道。
```

注意顺序语义：`numeric_limits.h:19-29` **使用** `half`/`bfloat16_t`，所以本头必须在
`common.h` 之前进入 TU —— `-include` 天然满足；若改成显式 `#include`，必须插在
`common.h:18`（`#include "tl_templates/ascend/numeric_limits.h"`）**之前**。

### 路 B：落进主仓模板树（P0-2 判决稳定后再做，需上游协作）

把文件重命名为 `src/tl_templates/ascend/port910b_compat.h`，在 `common.h` 的
`#if defined(TL_ASCEND_SIMT)` / `#else` 分支里 `#include` 它。
**P0-1 不动主仓**（纪律），故此处只登记方案，不落补丁。

### 目标代际开关（两条路都必需）

```bash
export ASCEND_NPU_ARCH=dav-2201      # tilelang/contrib/bisheng.py:93 读它，默认 dav-3510
                                     # 生效点：bisheng.py:113 -> --npu-arch=dav-2201
# 910B 面 = **不要**定义 TL_ASCEND_SIMT（现状：全仓无人注入该宏，默认即 910B 面）
# 950 面 = -DTL_ASCEND_SIMT=1（届时兼容层整体空转，已实测：见 §5 的 GATE 配置）
```

---

## 3. 开关清单（上卡迭代时只动编译器命令行，不动本头）

| 宏 | 默认 | 语义 | 什么时候改 |
|---|---|---|---|
| `TL_ASCEND_SIMT` | 未定义（=910B 面） | 总罩：定义后本头整体空转，950 真头接管 | 回 950 目标时 `-DTL_ASCEND_SIMT=1` |
| `TL_PORT910B_BF16_BUILTIN` | 自动探测 `__BFLT16_MANT_DIG__`/`__CLANG_BF16__` | 1 = `bfloat16_t` 直接 typedef clang `__bf16`；0 = 自带 storage struct（float 提升算术） | P9 若报 `__bf16` 不支持 → 显式 `=0`；若 bisheng 的 `BFloat16` 代理与 struct 冲突 → 试 `=1` |
| `TL_PORT910B_HALF_BUILTIN` | 自动探测 `__FLT16_MANT_DIG__` | 1 = `half` = `_Float16`；0 = 自带 storage struct | 同上，fp16 侧独立可切 |
| `TL_PORT910B_SKIP_<name>` | 未定义 | **逐名字退让**：定义了就让该名字归 bisheng，本头不再声明 | 报 `redefinition of 'float2'` → `-DTL_PORT910B_SKIP_float2`；`make_float2` 是 bisheng 内建（上游取证：`codegen_ascend.cc:2723-2725` "bisheng rejects them outside a VF body"）→ 大概率需要 `-DTL_PORT910B_SKIP_make_float2` 等 |
| `ASC_LOCK_BLOCK` / `ASC_LOCK_NON_BLOCK` | 0 / 1 | `asc_lock/unlock` 的模式实参 | 若 910B 真头已有枚举，`#ifndef` 自动让位，无需干预 |
| `__SIMT_DEVICE_FUNCTIONS_DECL__` | 空 | `ascend_fp8.h:9` 的 `TL_DEVICE` 前缀 | `#ifndef` 保护，无需干预 |

**冲突处置手册**（P9 的一次编译通常就能列全）：
- `redefinition of 'X'` → 加 `-DTL_PORT910B_SKIP_X`（X ∈ 附表 B1-TYPE/B2-CVT 的名字）。
- `unknown type name 'bfloat16_t'` → 兼容头没进 TU：查 `-I`/`-include` 是否落在 device 侧标志（不是 host 侧）。
- `use of undeclared identifier 'asc_*'` → 这是 **D2/C 类**，兼容层不承诺，转 `probe910b.sh P3` 的裁决。

---

## 4. 上卡执行顺序（P0-2 开局 15 分钟）

```bash
cd system-one/release/ascend/port910b
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_NPU_ARCH=dav-2201
./probe910b.sh P1 P2 P3      # ① 头存在性 ② 最小 TU ③ 24 个 D 类符号点名
./probe910b.sh P9            # ④ 兼容层形状在 bisheng 前端是否成立
# 拿到结果后：回填 INVENTORY.md §3-D 计数与 §4 墙0 判决，再决定 P1-1 是否开工
```

`probe.log` 原文必须贴回 run notes（域 `p2-13-ascend-runtime`），作为 D 类降级为 A/B/C 的凭据。

---

## 5. 本机可复跑的验证（离线三连）

```bash
cd release/ascend/port910b
./checklist.sh                 # audit：12 族穷举 → 反查清单，AUDIT PASS + 分类计数
./checklist.sh gap             # 族外残差体检（人工复核队列，见 INVENTORY §7）

clang++ -std=c++17 -I. host_selfcheck.cc -o /tmp/sc && /tmp/sc                    # auto
clang++ -std=c++17 -DTL_PORT910B_BF16_BUILTIN=0 -DTL_PORT910B_HALF_BUILTIN=0 \
        -I. host_selfcheck.cc -o /tmp/sc && /tmp/sc                               # storage
clang++ -std=c++17 -DTL_PORT910B_BF16_BUILTIN=1 -DTL_PORT910B_HALF_BUILTIN=1 \
        -I. host_selfcheck.cc -o /tmp/sc && /tmp/sc                               # builtin
clang++ -std=c++17 -DTL_ASCEND_SIMT=1 -fsyntax-only -x c++ port910b_compat.h      # 950 面应空转
```

主机自检覆盖的等价性（把 B 类判决从"推断级"抬到"实证级"，但**仅限主机前端**）：
`TL_BIT_CAST` 可用的 2 字节平凡类型、`(float)val` C 风格转换、`((bfloat16x2_t*)&v)[i].x`
lane 位视图、`((ushort2*)&v)[i].x`、`*(unsigned long long*)&make_float2(...)` 地址重解释、
bf16/fp16 的 RNE 数值（fp16 与 clang `_Float16` 神谕 20 万次随机对拍 mismatch=0）。

---

## 6. 本兼容层**不**承诺的事（勿越界引用）

1. **不提供任何 `asc_*`  intrinsic**。C 类 250 符号（SIMT 线程模型、SIMD 寄存器值方言、
   谓词寄存器、mx/fp8/hf32/atomic store-mode、nd2nz/dual-copy）必须由 `src/ascend/` 的
   lowering/codegen 改造解决，见 `INVENTORY.md §3-C`。
2. **不解决 `reduce.h` 漏罩缺口**。`common.h:19` 无条件 include `reduce.h`，而后者含 29 处
   `__simt_callee__`/`threadIdx`/`asc_shfl_xor`，910B 面上这条 TU 编不过——P2 探针会直接撞上它，
   修法属 C1 组（罩起来 + 新写单流 UB 归约），不是兼容层的一行 `#define`。
3. **不保证 910B 有 `__bf16`/`_Float16`**。所以两个后端都备好了，且 storage struct 后端零依赖。
4. **D 类 44 符号的存在性仍未裁决**。本头里的"910B 等价物"在 `probe910b.sh` 出结果前，
   一律按推断级看待。
