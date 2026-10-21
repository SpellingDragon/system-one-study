# RESULT.md — 执行代理 B / p2-13-ascend-runtime P1-1 波：add_ln + readout 的 910B DSL kernel

**判决环境**：容器 `cann910b-b`（CANN 8.5.0，bisheng/clang 15.0.5，`--npu-arch=dav-2201`，
pip tilelang 0.1.15，`/tilelang` 主仓只读挂载）。本地无 NPU ⇒ 验收上限＝
`tilelang.compile(..., target="ascend", out_idx=-1)` 成功（E2E-COMPILE PASS）。

**结论一句话**：两 kernel **均编译通过**（`B-ADDLN-COMPILE-PASS` / `B-READOUT-COMPILE-PASS`），
代价是给 `port910b_compat.h` 打了 4 处 GAP-B 补丁（G1 GM↔UB 搬运件错 arch、G2 跨 pipe 事件缺失、
G3 store L2 枚举缺失、G4 标量 `rsqrtf` 缺失）；compat 缺口全文见 `compat_gap_B.md`。
**主仓与 system-one 的 git/openspec 状态未动、未 commit。**

## 复现链（宿主机，四步；每条都是可直接粘贴的命令）

```bash
cd $REPO_ROOT/release/ascend/port910b/attempts/B
bash run_b.sh env_setup.sh 6          # [1] overlay 主仓模板→pip 树 + bisheng 注入自愈
bash run_b.sh apply_compat_gapB.py 11 # [2] GAP-B 补丁（幂等，备份 .preGapB）
bash run_b.sh e2e_cube_baseline.py 4  # [3] 判决环境自证：E2E-CUBE-ONLY PASS
bash run_b.sh b_addln_910b.py   14    # [4a] B-ADDLN-COMPILE-PASS
bash run_b.sh b_readout_910b.py 14    # [4b] B-READOUT-COMPILE-PASS
```

---

## ① 完成情况（逐条附凭据）

- [x] **add_ln 910B kernel 编译通过（静态形状）**
  验证：`bash run_b.sh b_addln_910b.py 14` →
  ```
  B-ADDLN-BEGIN [static rows=64 dim=512 bm=8 res_dtype=float32 out_dtype=float16 eps=1e-05 UB~28KB]
  B-ADDLN-GEN len=3001 miss=[] H/Y_refs=2 kernel='main_kernel' in src: True
  B-ADDLN-COMPILE-PASS
  ```
- [x] **add_ln 动态行数（`T.dynamic("rows")`）同样通过**
  验证：`bash run_b.sh b_addln_910b.py 6 --dyn` → `B-ADDLN-GEN len=3616 ... B-ADDLN-COMPILE-PASS`
- [x] **add_ln 位宽开关两形态通过**（与参照 `res_fp32/out_fp16` 同义）
  验证：`B_RES_FP32=0 bash run_b.sh b_addln_910b.py 3|tail -2` → PASS(len=3020)；
  `B_OUT_FP16=0 ...` → PASS(len=2995)
- [x] **readout 910B kernel 编译通过（scores = 末位行 × 字母行）**
  验证：`bash run_b.sh b_readout_910b.py 12` →
  ```
  B-READOUT-BEGIN [batch=8 seq=32 dim=512 k=26 bm=4 hid/weight=float16 scores=fp32 UB~60KB]
  B-READOUT-GEN len=1855 main_kernel=True uses_bulk_copy=False bounds_guard=True expf=False rsqrtf=False
  B-READOUT-COMPILE-PASS
  ```
- [x] **readout 形态矩阵**：`R_HDT=float32` → PASS(1843)；`R_BM=8 R_BATCH=32` → PASS(1856)；
  `R_HDT=bfloat16` → **FAIL `fatal error: error in backend: not support bf16 type cast`**（预期内，见 ②/5）
- [x] **GAP-B  compat 补丁落容器并自证**
  验证：`bash run_b.sh apply_compat_gapB.py 11` → `PATCHED ...(+3067 bytes)`，
  符号计数 `asc_sync_notify:3 asc_sync_wait:3 asc_store_l2_cache_mode:3 asc_copy_gm2ub_align:2
  asc_copy_ub2gm_align:2 #define asc_sync_notify:1 __cce_scalar::copy_ubuf_to_gm:1 float rsqrtf:1`
- [x] **补丁后 Cube 面不回归**：`bash run_b.sh e2e_cube_baseline.py 4` → `E2E-CUBE-ONLY PASS`
- [x] **"清缓存真编译"复验（避免 binary cache 假 PASS）**
  验证：`docker exec cann910b-b rm -rf /tmp/agentB/cache_dump && bash run_b.sh dump_final.py 12` →
  `B-ADDLN-COMPILE-PASS / SAVED /tmp/agentB/gen/final_addln.asc bytes=3001 /
   B-READOUT-COMPILE-PASS / SAVED .../final_readout.asc bytes=1855 / DUMP-FINAL-DONE`
  （生成码已回传本目录：`gen_addln.asc`、`gen_readout.asc`）
- [x] **910B vector 面构件可用性探针 7/11 PASS**
  验证：`bash run_b.sh probe_910b_faces.py 15 | grep -E "^P[0-9]|PROBE-DONE"` →
  `P0a P0b P2 P3 P4 P5 P10 = PASS`；`P6(exp) FAIL`、`P7(gemm 非 Cube) FAIL`、
  `P8(bf16 混排) FAIL`、`P9(T.Parallel) DIALECT-REJECT`；末行 `PROBE-DONE 7/11 PASS`
- [x] **G2 写法 A/B 取证（决定补丁形态）**：`bash run_b.sh probe_setflag.sh 40` →
  `V1 OK / V2 COMPILE-FAIL(the 1st parameter maybe need a type 'pipe_t') / V3 OK / V4 OK / V5 OK`
- [x] **标量数学件取证**：`bash run_b.sh probe_math_symbols.sh 30`（GAP-B 后）→
  `OK::rsqrtf OK::sqrt OK::rsqrt_alt MISSING::sqrtf expf exp logf log powf fabs floorf ceilf tanhf
   __nv_rsqrtf __nv_expf builtin_sqrt`
- [x] **bf16 崩点定责**：`bash run_b.sh dump_p8.py 26`（4 崩 2 通）+ `bash run_b.sh dump_bf16.py 55`
  → 原文 `fatal error: error in backend: not support bf16 type cast`
- [x] **T.gemm 非 Cube 定责**：`bash run_b.sh dump_p7.py 8` →
  `compat.h:555: error: function type 'void (__cc__ float*, __ca__ float*, __cb__ float*,
   unsigned short, ...)' of 'mad' does not support the given target feature`
  ⇒ 是 **Cube 件在 `__vector__` 子目标上不可用**（方言/子目标约束），非 compat 缺陷
- [x] 交付文档：`compat_gap_B.md`（G1–G4 补丁原文 + 原生件签名 + 未补清单）、本文件
- [ ] **数值对拍 / 上卡运行**：本地无 NPU，超出本次验收上限（任务书明示），**未完成**
- [ ] **readout 的 probs/softmax 入 kernel**：受 `expf` 缺失阻塞，见 ③/⑤，**未完成（有意留白）**
- [ ] P7 型 Cube 面 readout 投影（`T.gemm`+`with Cube()`）：未做（readout 走标量点积即可编译），**未验证**

## ② 错误与阻塞（含已解决的）

1. **`bundle_ondemand.sh [3]` 的注入把 `bisheng.py` 改成 SyntaxError**
   `SyntaxError: unterminated string literal (line 111)` @ pip `tilelang/contrib/bisheng.py:111`
   → 处置：**已修复**（`fix_bisheng.py`：正则撤销坏注入 → 以"追加列表元素"重注入 → `ast.parse` 自证）。
   根因是通用缺陷：锚 `'"-O2", "-fPIC", "-std=c++20"'` 在 0.1.15 **不是列表结尾**，注入串却硬加 `]`。
   连带坑：修复器自身**不能 `import tilelang`**（会加载坏 bisheng.py 而自举失败），改走 `sys.path` 探测。
2. **GAP-B 第一版的 `asc_sync_notify/wait` 写成 inline 函数 → 编译 FAIL**
   `error: the 1st parameter maybe need a type 'pipe_t'` @ compat（G2 定义行）
   → 处置：**已修复**（改函数形宏，见 ① 的 probe_setflag A/B）。教训：CCE builtin 要字面枚举常量，
   而 toolchain 自己的 `__cce_set_flag` 靠丢弃实参硬编码 pipe 对来"编过"，**照抄它会得到错误的同步语义**。
3. **`T.Kernel(..., threads=)` 被拒**（早期 9 项探针全 FAIL）
   `Kernel() got an unexpected keyword argument 'threads'` @ `tilelang/ascend/language/kernel.py:139`
   → 处置：**已绕行**（Ascend 方言用 `T.Kernel(1-D grid)`；`threads=` 只属 common 方言）。
4. **`expf` 无标量形态**（P6 FAIL，原文 `use of undeclared identifier 'expf'`）
   → 处置：**未绕行**（补近似或改向量件都是数值语义改动，代价大且无卡不可验）→ readout 只出 scores，
   probs 留 host；已列入 ⑤ 待裁决。
5. **bf16 后端缺陷**：`error in backend: not support bf16 type cast`（exit code 70，前端崩溃）
   → 处置：**绕行**（两 kernel 的激活位宽钉 fp16；生产口径本就是 fp16：`model.py:329 .half()`、
   `parity.py:136 h.to(torch.float16)`）。compat 无法补：`bfloat16_t = __bf16` 是内建类型。
6. **`AscendBinaryCache` 命中导致 dump 工装空手而归**（两次 `NO-GENCODE`）
   @ `tilelang/ascend/backend.py:24-38`（key 不含模板内容，命中即返回缓存 ELF 不调 bisheng）
   → 处置：**已修复**（`dump_final.py` 设一次性 `TILELANG_CACHE_DIR` + 把 `AscendBinaryCache.load`
   打成恒 None ⇒ 强制真编译）。**附带风险提示**：改 compat 后若不清缓存，判决会失真。
7. **探针错误提取抓到 warning 而非 error**（早期把 `kernel_log.h ... warning: inline function
   'AscendC::AssertImpl<...>' is not defined` 当首错误，掩盖真判决）
   → 处置：**已修复**（收紧为 `": error:"`；重跑得 7/11）。
8. **容器公共 `/tmp` 同名文件属主 uid 501，覆盖被 Permission denied**（并发代理互踩）
   → 处置：**已绕行**（专用 scratch `/tmp/agentB/`，`run_b.sh` 内置 mkdir）。
9. **数学件批量 TU 结论失真**（`-ferror-limit=0` 下只报 rsqrtf 漏 expf，伴
   `Fatal Err ... Diagnostic info can not be reported correctly`）
   → 处置：**已修复**（一符号一 TU 隔离，`volatile` 防常量折叠）。
10. **环境类**：本地无 NPU（不能 run/对拍）；容器内无 `/tilelang` 之外的可写主仓副本；
    `bisheng` 需显式 `BISHENG_HOME=<...>/ccec_compiler` 否则走外层 wrapper 的 `-x cce` 形态。

## ③ 疑惑点与自行决策（均为"没问但自己定了"的事）

1. **readout 的范围界定**：任务书写"取序列末位/读出行投影 logits"。我按
   `sys1/decision/readout.py::option_scores`（:90-114）实现 **scores (B,k)**，
   **不含 `to_probs`/softmax**——因为 910B 标量面没有 `expf`（实测），且 `T.Parallel` 外圈被方言拒。
   依据：`Readout.scores` 与 `probs` 是两个字段，`option_scores` 才是"唯一碰模型产物"的那一步。
   ⇒ 若编排者要求 probs 也在 kernel 内，需裁决 ⑤-1。
2. **pos 作为 kernel 输入**：`last_positions()`（readout.py:71-87，含长度合法性校验）留在 host，
   kernel 只吃 `POS int32 (B,)`。依据：那是纯标量校验逻辑，塞进 kernel 只会引入不可验证的分支；
   且 `option_scores` 本身也是在 host 算 pos 再 gather。
3. **激活位宽取 fp16 而非 bf16**：见 ②-5。同时 readout 的 `R_HDT`、add_ln 的
   `B_RES_FP32/B_OUT_FP16` 都留了开关，编排者可一条命令改口径复测。
4. **add_ln 的三趟改为"一次 gm→ub + UB 内三扫"**：参照是每趟直读 GM（Metal 无 UB 概念）。
   数值口径**逐字不变**（同一批 `cast(x)+cast(r)`），改的是访存形态；理由：Ascend 面逐元素读必须
   先入 UB（`T.copy`），三趟各读一遍 GM 会把 DMA 翻三倍。风险：UB 占用（bm=8,dim=512 约 28KB）。
5. **形状取静态 + 另测动态**：默认静态 `rows=64/dim=512/bm=8`（与参照 `ROW_BLOCK=16` 不同，
   见 ④-4），并额外证明 `T.dynamic("rows")` 在 Ascend 方言可编（①-2）。行数分块常量没做成动态符号，
   避免一次改动引入两个未验证维度。
6. **readout 全用标量散读，不用批量 `T.copy`**：gather 语义（按 `pos/ids` 散取行）本就不是连续块，
   且顺手绕开 GAP-B 搬运件的 32B 粒度契约（`k=26` 个 int32=104B 会被 `/32` 截断）；
   codegen 自动补的界守卫（`gen_readout.asc:11,22`：`0 <= POS[...] < 32`、`0 <= IDS[t] < 4096`）
   实测生成，越界读回 0 而非崩，安全性可接受。
7. **`rsqrtf` 用 `1.0f/sqrt(x)` 而非向量近似件**：编译面等价、语义保守；已在 compat 注释与
   `compat_gap_B.md §4` 明写"不等价于 vrsqrt，上卡后可能有 ulp 级差异"。
8. **补丁只打容器 pip 副本**：主仓是模板真源且被禁改；因此 `compat_gap_B.md` 里贴了可直接
   回合到主仓的**补丁原文**（含锚点说明），供编排者决定是否上主仓。
9. **未 commit、未动 openspec/runs**：硬边界。`git status` 只在 system-one 侧新增了 attempts/B/
   下我的文件（未 add）。

## ④ 偏离记录（与参照/spec 不一致之处）

| # | 计划/参照原文 | 实际做法 | 理由 |
|---|---|---|---|
| 1 | `add_ln_mps.py:62` `with T.Kernel(ceildiv(rows,bm), threads=128)` + `if tx == r` | `with T.Kernel(ceildiv(rows,bm)) as bx` + 块内 `T.serial(BM)` 逐行 | Ascend 方言 `T.Kernel` **没有 threads 形参**（kernel.py:139），非 SIMT 面无 thread binding |
| 2 | 参照逐元素 `X[row,j]`（三趟各读一遍 global） | `T.copy` 入 UB，三趟扫 UB | Ascend vector 面逐元素读必须经 UB；DMA 减为 1/3 |
| 3 | `T.Parallel`（示例 `example_rmsnorm.py` 用法） | 全 `T.serial` 嵌套 + 纯标量 store | 任务书预警 1/2 已复现：`P9 DIALECT-REJECT: T.Parallel loop must be inside a VF block`；vector store 撞 `ScalarDcacheBypass` |
| 4 | 参照 `ROW_BLOCK=16` | 默认 `bm=8`（add_ln）/ `bm=4`（readout） | UB 预算与 32B 粒度；bm=8 时 8192B/16384B 全是 32 倍数（`gen_addln.asc:12-13` 实证），bm 也可 16，非语义偏离只是配置 |
| 5 | readout 参照含 `probs`（`Readout.probs`） | 只出 `scores` | `expf` 缺失（②-4），非本次可交付 |
| 6 | compat 原 10 参 `asc_copy_ub2gm_align` 签名 | 按 codegen 实发 **7 参**重写 | codegen 与 compat 本来就不一致（`codegen_ascend.cc:1405-1433` 为权威），改后 P2/P5 PASS |
| 7 | 用 `__cce_set_flag/__cce_wait_flag`（compat 里"看着像"的件） | 用宏转 `__cce_scalar::set_flag/wait_flag` | 前者丢弃 pipe 实参硬编码 M→V，同步语义错误 |

## ⑤ 需编排者裁决 / 产物清单

**待裁决（按优先级）**

1. **`expf` 一类标量数学件缺失**：readout 的 probs 与一切带激活/对数的 vector 面算子都会被挡。
   选项：① probs 留 host（我当前做法）；② 让 DSL 在 910B 上发射 VF 块用 `__cce_scalar::vexp`
   （需先定"910B 面怎么写 VF 块"，`T.SimdVF/T.SimtVF` 是否属 950 面需澄清）；③ compat 多项式近似（需数值口径）。
2. **bf16 面在 910B 后端不可用**（`not support bf16 type cast`）：若某算子口径必须 bf16，
   需决定"uint16 搬运 + 手工位拼接"还是"该算子走 Cube 面"。
3. **GAP-B 是否回合主仓 compat**：我只改了容器副本。补丁原文在 `compat_gap_B.md §1`，
   其中 G1/G3 是**修正 codegen↔compat 契约不一致**（ub2gm 7 参 vs 10 参、3 参 config 形态错 arch），
   属真缺陷；G2 宏与 G4 shim 带数值/风格口径，需评审后再入仓。
4. **`bundle_ondemand.sh [3]` 注入缺陷**（对 0.1.15 通用失效，会直接崩一键上机）：
   建议替换为 `fix_bisheng.py` 的"追加列表元素 + ast 自证"形态。
5. **`AscendBinaryCache` 的 key 不含模板头内容**：改 compat 后不清缓存 = 判决失真。
   建议在 port910b 判决流程里固化"清 `TILELANG_CACHE_DIR` + 禁 binary cache"这一步（我有脚本）。
6. **GAP-B 搬运件的 32B 粒度契约**：非 32 倍数长度会静默截断。选项：codegen/host 补 pad /
   shim 内标量兜底 / 编译期硬报错（我倾向前两者之一，但无卡不可验证，故未动）。

**产物清单（全部在 `release/ascend/port910b/attempts/B/`；主仓与 system-one 其它路径零改动）**

- 交付 kernel（可直接执行）：`b_addln_910b.py`、`b_readout_910b.py`
- 交付文档：`compat_gap_B.md`（G1–G4 补丁原文 + CCE 原生件签名 + 未补清单）、`RESULT.md`（本文件）
- compat 补丁执行器：`apply_compat_gapB.py`
- 环境/执行工装：`env_setup.sh`、`fix_bisheng.py`、`run_b.sh`、`probe_env.sh`
- 取证脚本：`probe_910b_faces.py`、`probe_setflag.sh`、`probe_math_symbols.sh`、`probe_cce_arch.sh`、
  `dump_p8.py`、`dump_bf16.py`、`dump_p7.py`、`dump_one.py`、`dump_generated.py`、`dump_final.py`
- 生成码留证：`gen_addln.asc`（3001B）、`gen_readout.asc`（1855B）
- 基线自证：`e2e_cube_baseline.py`（上层 `e2e_cube.py` 副本，`E2E-CUBE-ONLY PASS`）
- 容器内改动（不进版本库）：pip `tilelang/src/tl_templates/ascend/port910b_compat.h`
  （+3067B，备份 `port910b_compat.h.preGapB`）、pip `tilelang/contrib/bisheng.py`
  （注入 3 个编译选项，备份 `bisheng.py.bak910bB`）、scratch `/tmp/agentB/`

## ⑥ 硬边界自证（命令 + 实际输出）

```bash
cd $WORKSPACE_ROOT/tilelang && git status --porcelain | head
#  M src/tl_templates/ascend/common.h / dcache_bypass.h / debug.h / numeric_limits.h
# ?? src/c_api/  src/simt_api/  src/tl_templates/ascend/port910b_compat.h  tilelang.code-workspace
cd $REPO_ROOT && git status --porcelain | head
#  ?? release/ascend/port910b/attempts/
```

判定：**主仓那些 M/?? 条目全部先于本会话存在**（`stat -f "%Sm %N"` 实测 mtime：
common.h 10-08 00:40、debug.h 01:56、dcache_bypass.h 01:46、port910b_compat.h 11:38，
而本会话起始于 10-08 23:45 之后）⇒ 我未写主仓一个字节。
system-one 侧只有我写入白名单目录 `release/ascend/port910b/attempts/B/`（未跟踪），
**无 tracked 文件改动、无 git add / commit、未碰 openspec/runs、未碰远程/卡/OBS**。
容器侧改动仅限 `cann910b-b` 的 pip 运行副本与 `/tmp/agentB/`（overlay fs，不影响其他代理容器）。
