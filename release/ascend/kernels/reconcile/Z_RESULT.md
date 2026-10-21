# Z_RESULT.md — p2-13 P1-1k：§12 成套修正落地（cube intrinsic 路根因修复）

> 权威依据：`attempts/X_sec12_proposal.md`（逐槽 diff + 参数字典）× `attempts/X_FRACTAL_EVIDENCE.md`（§7 逐槽对照 / §8 U1–U6）。
> 定位：performance-track（cube intrinsic 路）根因修复；**不改产品标量路、不阻塞 P2**。

## 0. 一句话结论

**P0/P1/P2 三段成套落地 trunk §12，compile 面全绿、无回归；数值待下窗真机 E-1/E-2/E-3。**
- 写侧 `asc_copy_gm2l1_nd2nz`：8 参裸 `copy_gm_to_cbuf`（ND 线性块）→ 11 参官方 ND2NZ 件（按 dtype 选族）+ `gCol>=UINT16_MAX` 逐行拆趟兜底。
- 读侧 `asc_copy_l12l0a/b`：sid 槽不再吃 IR `m_start`（钉 0=P0）、`repeatTimes←kStep`、二维起点折进 src 指针、M 向外层 for 拆趟、`kStep==1` 退化支。
- U2/U3（dstGap 语义、L0 目的步进）与 `_transpose`/dn2nz 路 → **保持现状 + 显式 VERIFY U2/U3/U5**，未臆造修正。

## 1. 四条验证（均可复跑）

| # | 判据 | 原文/命令 | 结果 |
|---|---|---|---|
| ① compile 自证（提案） | `X_probe_proposal.py` 8/8 rc=0 | `docker exec cann910b-k bash -c 'source .../set_env.sh; cd /tmp && python3 X_probe_proposal.py'` | **8/8 PASS** |
| ①′ compile 自证（落地件·新） | `Z_probe_sec12.py` 8/8 rc=0（真件名/真签名，含三 dtype 族 + gCol 兜底 + 拆趟 + kStep=1 退化 + dn2nz 委托） | `docker cp Z_probe_sec12.py cann910b-k:/tmp/ && docker exec cann910b-k bash -c 'source .../set_env.sh; cd /tmp && python3 Z_probe_sec12.py'` | **8/8 PASS** |
| ② 九件无回归 | `compile_all_asc.py --target ascend` → `PASS=9 FAIL=0`，逐件 .o 字节数与改前基线**逐字一致**（2850/2250/3625/2106/2911/4003/1864/922/2842） | `docker cp 新compat → pip; docker exec cann910b-k bash -c 'source .../set_env.sh; cd /work && python3 ascend/compile_all_asc.py --target ascend'` | **PASS=9 FAIL=0** |
| ③ 产品/宿主面无回归 | host `.venv` `pytest tests -q -m "not integration"` → **332 passed**；`S_cpu_oracle.py --cpu`→S-DW-CPU-ORACLE-PASS rc=[0,0]；`U_oracle.py --prod`→PROD-BODY-ORACLE-PASS；Y 三件 `--only *_prod --track cpu-alias`：linear rel=2.54e-07 / dw rel=2.21e-07 / readout bitexact 全 PASS | `cd release && .venv/bin/python -m pytest tests -q -m "not integration"`；余见 §3 命令 | **全绿** |
| ④ intrinsic 轨编译判决 | `run_numerics.py --only gemm_l1_intrinsic/dw_intrinsic --track card --no-run` → 两条 `NUM-*-PASS rel=n/a (compile-only) track=performance` EXIT=0 | `docker exec cann910b-k bash -c 'source .../set_env.sh; cd /work/ascend/port910b && python3 run_numerics.py --only gemm_l1_intrinsic --track card --no-run'`（dw 同） | **compile-PASS**（数值未判，见 §4） |

> **未宣称数值已修**：①/①′ 只证 2201 语法/sema 面可编；④ 只证 compile-only。数值真机判据（NZ 位型 E-1、mad 对拍 E-2、sid 触发者二选一 E-3）随下窗。

## 2. 逐段 diff 要点（对提案行号引用）

### 2.1 P1 写侧 `asc_copy_gm2l1_nd2nz`（提案 §2；trunk `:894-941`）
- 弃 `copy_gm_to_cbuf` 8 参（= `DataCopyGM2L1Impl`，ND 线性块，不产 NZ）→ 按 `sizeof(ST)` 选 `copy_gm_to_cbuf_multi_nd2nz_b8/b16/b32s`（11 参，sema D1/D5/D6 rc=0）。
- 字段单位照抄官方 `kernel_operator_data_copy_intf.h:44-56` + 地址模型 `kernel_check_data_copy_overflow.h:505-520`：`nValue/dValue/srcNdMatrixStride/srcDValue`=**元素**，`dstNzC0Stride/dstNzNStride`=**32B 块**，`dstNzMatrixStride`=元素；`DEFAULT_C0_SIZE=32`。
- 取值照抄 `data_copy_wrapper_nd.h:74-97`：`ndNum=1 / srcNdMatrixStride=0 / dstNzC0Stride=Ceil(nRows,16)*16 / dstNzNStride=1 / dstNzMatrixStride=0 / sid=0`（`data_copy_impl.h:275` 字面 0）。
- IR 口径：`rowBytes=args[3]=src_row_stride_bytes`（GM 全行宽 = 官方 gCol）、`nRows=args[5]=n_value`、`cols=args[6]=d_value`（`copy.cc:401-410` + `codegen_ascend.cc:1458`）。
- **新增 `gCol>=UINT16_MAX` 逐行兜底**（提案 §2 末"须补守卫"，实现落 `wrapper_nd.h:99-107` 同款）：M 向逐行 `nValue=1 / srcDValue=width`（gCol 不再进任何 u16 槽 → 无截断），dst 每趟 +1 个 32B 块、src 每趟 +gCol 元素。
- VERIFY U1：`dstNzC0Stride(32B块)` 与读侧 `srcStride(row16块)` 是同一物理量两单位；`§12` 感知不到 codegen extent → 以注释钉死不变式 `L1 outer1 == Ceil(nRows,16)`，随 E-1 位型回读闭合。

### 2.2 P0+P2 读侧 `asc_copy_l12l0a`（提案 §1/§3；trunk `:969-995`）
- 形参 `sid` → **`mStart` 更名**（错位放大器，§7.2 末）。
- **P0 止血并入 P2**：第 7 槽 sid 钉 `(uint8_t)0`（sema [0,15]，运行期无保护 → m_start>15 = 507015 头号嫌疑，§2.4/§7.2）。P0-only 对照件保留在 `X_probe_proposal.py::P0_bleed_sid0`，供下窗 E-3 二选一。
- **P2 V1 降级四条硬规则**（`load_to_l0a_load2d.h:41-110`，§5.2）：① 二维起点折进 src 指针、`startIndex≡0`；② `repeatTimes=kStep`（u8，旧 `(uint8_t)mStep` 轴选反且 m_step>255 静默截断）；③ `srcStride`=row16 块数（口径本就一致）；④ M 向外层 for 拆 mStep 趟，每趟 src +1 分形块 / dst +kStep 分形块。
- `kStep==1` 官方同款退化支（repeat 沿 M、srcStride=1）。
- **U2/U3 保持现状 + 显式 VERIFY**：`dstStride` 丢弃并注 `U2`（假定 L0 紧凑 pitch==kStep，`CUBE_MAX_SIZE` 口径未抓到）；`dstGap≡0` 注 `U3`（L0A=0/L0B=nFraC0-1 未定论）。

### 2.3 P2-B 读侧 `asc_copy_l12l0b`（提案 §3；trunk `:996-1010`）
- 同 A 式 V1 降级：`startIndex≡0` 折指针、`repeatTimes=kStep`、`srcStride` 沿用、sid=0、transpose=false。
- **U3 保持现状 + 显式 VERIFY**：B 侧本轮 K-major 单趟形态，`(void)mStart;(void)mStep;(void)dstStride`；官方 L0B 转置由 L1 写侧 dn2nz 承担而非 load 位（`load_to_l0b_basic.h:111-119`），N 向拆趟与 dstGap 真值留 U3/E-4。

### 2.4 保持项（提案"不改项"，均落显式 VERIFY）
- `asc_copy_gm2l1_dn2nz`：保持委托（dn 专属件 `copy_gm_to_cbuf_multi_dn2nz` 与 nValue/dValue 互换关系未取证）→ `VERIFY U5`。
- `asc_copy_l12l0a_transpose` / `_b_transpose`：**代码一字未动**，仅旧 `V2/V2'` 标记改指 `U5/E-5`（transpose 件双 cfg 字族属、位序未取证）。
- `asc_set_gm2l1_nz_para`（空实现）：保持空实现（c0_stride 已由 P1 写侧自算；改用 codegen args[10] 属接口扩张需 codegen 同窗，本窗禁改）。
- `asc_fill_l1` / `asc_copy_l0c2gm`(V3) / `asc_mmad`：一字未动。
- §12 头部 `V1/V2/V3` 说明更新为定论口径 + `U1–U6` 指向 §8（避免 VERIFY 语义漂移）。

## 3. 疑惑点与自行决策（申报）

- **多候选变更无法区分的规避**：`openspec list` 顶层无 `p2-13` 条目（它在 `teacher-p2-production-full/changes/p2-13-ascend-runtime/tasks.md` 孙层），且 CLI `status --change p2-13` 报 not found。→ 按任务书权威文件名 `tasks.md:21` 直接定位孙任务，未强走 `openspec instructions apply`（schema 未铺该孙层）。**假设：P1-1k 即任务书 §"权威依据" 所指孙任务**。
- **新增 `Z_probe_sec12.py`（白名单允许的新建 attempts）**：X 探针编的是"提案粘贴文本"，为证**落地后真实 §12 件**（经 `common.h` 包含链 + 真件名/真签名），另写落地件探针，覆盖三 dtype 族 + gCol 兜底支 + 拆趟支 + kStep=1 退化支 + dn2nz 委托，8/8 rc=0。
- **Z 探针 gCol 兜底支用 `rowBytes=131072`（bf16）逼 `gColW=65536>=UINT16_MAX`**：单测兜底分支持续可编，不影响生产件（生产件走 `compile_all_asc` 判）。
- **无 `.venv` 在仓根**：host 轨均在 `release/.venv`（`S/Y/U_RESULT.md` 口径），已按此跑。

## 4. 性能预期与"数值待下窗 E-3"声明

- **性能预期**：P1 单发支（gCol<65535）= 1 条 ND2NZ 指令，与旧 8 参同量级；gCol>=UINT16_MAX 逐行兜底 = nRows 条指令（纯性能面，K 极宽 tile 才触发，语义无损）。P2 读侧 M 向拆趟 = mStep 条 load（旧形单条但槽位非法）；kStep==1 退化支仍 1 条。整体相对旧"错形"是**从非法访问变合法**，指令数上升属结构性代价，非回退。
- **数值未判**：三处 compile 全绿只证 2201 语法/sema 面可编（R7/R19），**不证数值**。数值判据留待真机最小判别实验：E-1（GM→L1→GM 位型证 NZ）、E-2（16³ mad 对拍收敛 U1/U2/U3）、**E-3（P0-only sid 钉 0 判别 507015 触发者）**、E-4（L0 步进/dstGap 标定）。**禁止只落 P1 或只落 P2**——本窗已成套，未犯此禁。

## 5. 产物清单

- **修改**：
  - `src/tl_templates/ascend/port910b_compat.h`（trunk §12，1062→1167 行；md5 `f56432b0...`）
  - `release/ascend/port910b/patches/port910b_compat.h`（overlay 镜像，md5 与 trunk 一致）
  - `release/ascend/.../p2-13-ascend-runtime/tasks.md`（第 21 行 P1-1k 勾 `[ ]`→`[x]`，未动他人行）
  - 容器 pip `.../tilelang/src/tl_templates/ascend/port910b_compat.h`（docker cp，md5 一致）
- **新建**：
  - `release/ascend/port910b/attempts/Z_patch_sec12.py`（定点替换脚本，锚点唯一性断言）
  - `release/ascend/port910b/attempts/Z_probe_sec12.py`（落地件 compile 自证探针，8/8 rc=0）
  - `release/ascend/kernels/reconcile/Z_RESULT.md`（本件）
- **禁改项复核**：`gemm.h`/`common.h`/`debug.h`/`dcache_bypass.h`/`numeric_limits.h`、`ascend/kernels/*`、`run_numerics.py`/`compile_all_asc.py`/`card_run.sh`、`src/ascend/**`、tests —— **一字未动**；无 git 写。
