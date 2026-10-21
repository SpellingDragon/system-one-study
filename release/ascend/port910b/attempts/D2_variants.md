# D2 — §12 分层判别器（E-1..E-4 最小判别实验）：变体说明 + 单窗四趟配方

**变更**：p2-13 / P1-1n（执行代理 D2）  **日期**：2026-10-11
**落点**：trunk `tilelang/src/tl_templates/ascend/port910b_compat.h` §12（唯一可改件；仅 §12 区）
**头文件指纹**：`md5 ca023c2e01746ca52c87f36bcea93ee9`，1248 行，`#if*=93 / #endif=93`、嵌套终深度 0
**三处同步**：trunk == `release/ascend/port910b/patches/port910b_compat.h` == 容器 pip 树（同一 md5）
**一句话**：三个正交的编译期开关把 X 设计的 E-1/E-2/E-3/E-4 做成**默认全 OFF、可单窗归因**的判别载荷；
离线已证「默认路径与 A2 基线逐字节等价」+「A/C 确有判别力、B 在本窗两腿是空开关」——
**B 的空开关本身就是 E-3 的结案证据，不用花卡**。

---

## 1. 三个变体是什么

| 开关 | 位点（本文件行号） | 退回的旧形态 | 回答的问题 | 件覆盖 |
|---|---|---|---|---|
| `-DTL910B_VARIANT_A` | 940–997（`asc_copy_gm2l1_nd2nz` 体内：`#if`940 / `#else`953 / `#endif`997） | GM→L1 走 **8 参 `copy_gm_to_cbuf` ND 线性块拷贝**（= 官方 `DataCopyGM2L1Impl`，`kernel_operator_data_copy_impl.h:83-108`），丢 IR `d_value`、无 NZ c0 pitch | **E-1 / 裁决②**：L1 里是 ND 还是 NZ、"ND vs NZ 布局"是不是主因 | `asc_copy_gm2l1_nd2nz` + 委托它的 `asc_copy_gm2l1_dn2nz`（995 行委托 ⇒ 同开关自动覆盖） |
| `-DTL910B_VARIANT_B` | 宏块 887–895（注释块 871–886），生效于 5 个 sid 槽位点 1038 / 1054 / 1061 / 1076 / 1089（`asc_copy_l12l0a`、`l12l0b` 的默认支与 C 支） | 读侧第 7 槽（4-bit `sid`）**重新吃 IR `mStart`**（Z 的 P0 钉 0 之前行为） | **E-3 / 裁决③**：507015 是否 sid 越界触发 | 仅 `asc_copy_l12l0a` / `asc_copy_l12l0b`（非转件）。`*_transpose` 件**无 sid 槽、也无任何变体分支**（实测：`asc_copy_l12l0a_transpose` 与 `l12l0b_transpose` 两个件体内 `TL910B_L1_SID` 计数=0、`TL910B_VARIANT` 计数=0） |
| `-DTL910B_VARIANT_C` | 1029–1064（A 侧）、1069–1091（B 侧） | 读侧退**单条指令**：`repeatTimes←mStep`（轴选反，应 `kStep`）、`startIndex←kStart`（**不折指针**）、`dstGap←dstStride`、`(void)kStep`、**无 M 向拆趟** | **E-2 / E-4（U1 块线性序 / U2 L0 目的步进 / U3 dstGap 真值）** | 同 B：仅非转件 |

行号证据（写进头文件注释里的原证）：A 引 `X_FRACTAL_EVIDENCE` §2.5 D7 + §6 证据 1/2/4；
B 引 §7.2 表第 7 行 + §2.2 S1/S2 + §2.4 RT3；C 引 §7.2 逐槽错位表（`〔3〕〔4〕〔6〕`三槽）。

### 1.1 覆盖关系 = 判读前置（读错会得出相反结论）

`dw_intrinsic` 的生成码里，读侧**只**出现 `asc_copy_l12l0a_transpose` / `l12l0b_transpose`
（实参 `0, 0, 4, 8, 4, 8`；从 A2 基线缓存 `device_kernel.asc` 直接读出），而 `_transpose` 件
既无 sid 槽、也从未走过 P2 的折指针/拆趟（Z 未改该件，§8-U5）⇒

| | `gemm_l1_intrinsic`（`main_kernel`） | `dw_intrinsic`（`dW_l0tr_kernel`） |
|---|---|---|
| **+A** | **活**（写侧 nd2nz） | **活**（写侧 nd2nz） |
| **+B** | **空**（`gemm.h:68/74` 给 mStart 传**字面 0**） | **空**（只发 `_transpose`，无 sid 槽） |
| **+C** | **活**（模板内非转件 l12l0a/l12l0b） | **空**（只发 `_transpose`） |

⇒ **dw 腿只对 A 有反应**。凡在 dw 腿看到 "+C/+B 与基线同色"，那是**覆盖关系**，不是"变体失效"，
也不能当成"读侧被排除"的证据（读侧的证据只在 gemm_l1 腿）。

---

## 2. 注入面：为什么不需要改任何禁改件

开关只需把 `-DTL910B_VARIANT_x` 送进 bisheng 的设备编译命令行。tilelang 侧**唯一**的命令行生成函数是
`tilelang/contrib/bisheng.py::get_bisheng_compile_options(npu_arch, options)`（:105，`patch_bisheng.py`
写死 `-DTL_PORT910B_NATIVE_TYPES` 与两个 `-I` 的那一行就在它 :111）。三处调用点全部**运行时取属性/全局名**：

* `tilelang/ascend/backend.py:23` → `bisheng.get_bisheng_compile_options(target_arch, options)`
* `tilelang/contrib/bisheng.py:187` → 同模块全局名
* `tilelang/jit/adapter/libgen.py:145,161` → 函数内 import

⇒ **monkeypatch 这一个函数即可覆盖全链**，`D2_probe_legs.py` 就是这么做的：
不改 `patch_bisheng.py`、不改 `card_run.sh`、不改任何 `*_asc.py`/`*_kernel.py`/`compile_all_asc.py`/
`run_numerics.py`/`ascend_env.py`。

顺带的**缓存安全**：`AscendBinaryCache.make_key` 把 options 计入键 ⇒ 换 `-D` 自动换缓存条目，
不会撞旧 aibin 拿假绿。`D2_probe_legs.py` 另给每组独立 `TILELANG_CACHE_DIR=/tmp/d2_legs_<combo>`
双保险（可整组留档/复核）。

⚠ **不要用 `TL_DEVICE_COMPILE_FLAGS`（`tl.device_compile_flags`）这条 env 路**：`patch_bisheng.py`
的 result 行在 `normalize_options(options)` **之前**写死，且 env→PassConfig 的传导要过 tvm 配置层，
窗内多一个不确定因素；monkeypatch 点唯一、可打印、可自检（每组开跑前打印
`## inprocess combo=A flags=['-DTL910B_VARIANT_A']`，**没这行 = 开关没注入 = 该趟作废**）。

---

## 3. 单窗四趟 SOP（可直接逐行粘贴）

**先就位三样东西**（缺一不可，否则四趟全是假绿）：
(1) 卡上 tilelang 安装树里的 §12 头文件 = md5 `ca023c2e…`（见 §8.1）；
(2) `attempts/D2_probe_legs.py` 已拷到卡上工作目录；
(3) CANN `set_env.sh` 已 source（bisheng 在 PATH 上）。
下面第 0 步用 `$TL_SRC` 指代卡上 `…/tilelang/src/tl_templates/ascend`，由命令自己算出来。

```bash
# ── 0) 开窗后第 1 分钟：先验"开关在不在场"（30s，不做这步四趟全是假绿）──────────
TL_SRC=$(python3 -c "import tilelang,os;\
print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
md5sum $TL_SRC/port910b_compat.h            # 期望 ca023c2e01746ca52c87f36bcea93ee9
grep -c  TL910B_VARIANT $TL_SRC/port910b_compat.h   # 期望 14（三开关的分支/注释行总数）
grep -c 'TL910B_L1_SID(mStart),' $TL_SRC/port910b_compat.h  # 期望 5（sid 槽位点）
```

```bash
# ── 1) 四趟（同一进程组、每组独立缓存目录、编译面自动 --no-run）──────────────────
#     先离线自检一次（不带 --card，只编不跑，验证注入链与指纹；≤30s）
python3 D2_probe_legs.py --combos default,A,B,C --baseline <A2基线缓存目录>

#     真机四趟（--card ⇒ 去掉 --no-run，跑数值 + 落 npz/rel 判据）
python3 D2_probe_legs.py --combos default,A,B,C --card \
        --baseline <A2基线缓存目录> 2>&1 | tee /tmp/d2_window_$(date +%H%M).log
```

每组内部实际跑的是（`D2_probe_legs.py` 自己起干净子进程并注入 `-D`）：

```
default 组：compile_all_asc.py --target ascend                # 九件回归门（不吃 §12）
          + run_numerics.py --only gemm_l1_intrinsic --track card
          + run_numerics.py --only dw_intrinsic      --track card
A/B/C 组 ：只跑后两条 §12 腿（九件腿与 §12 无关，重复跑是浪费卡时）
```

**每趟必须留下的三个判决行**（缺一即该趟不计入归因）：

| # | 判决行 | 从哪读 |
|---|---|---|
| 1 | 注入确认行 `## inprocess combo=X flags=['-DTL910B_VARIANT_X']` | 组日志首行 |
| 2 | 两腿数值/错误：`NUM-gemm_l1_intrinsic-PASS rel=…` / `NUM-dw_intrinsic-PASS rel=… dir_disc=…` 或 `…-FAIL <错误码原文>` | run_numerics 判决行（**记下 507015 之类的错误码原文与 rel 数值**，红-红之间要靠数字区分） |
| 3 | 机器码面指纹 `so=DIFF/SAME` + `aibin 集合交集 k/n` | 组尾结论表（证明"这趟真的换了码"，防"开关没生效的同色"） |

**停窗条件**：四趟若任一趟缺判决行 1 或 3（= 注入没生效/码没变），**先修注入链再重跑该趟**，
不要在窗内改 §12 代码；若四趟全跑完但 dw 腿在 +C 趟与基线**不同色**，说明覆盖关系模型错了
（`_transpose` 件被谁改了？）——立刻停手回报，这比任何单趟结果都重要。

---

## 4. 估时与预算（实测/估计分列）

| 段 | 耗时 | 来源 |
|---|---|---|
| 编译面 default 趟（九件门 + 2 条 §12 腿） | **12 s** | 容器 `cann910b-k` 实测（`SECONDS` 差值，CANN 8.5.0 / bisheng 15.0.5） |
| 编译面 A / B / C 趟（各 2 条腿） | **3 / 2 / 2 s** | 同上 |
| 四趟全跑（编译面，`--combos default,A,B,C`） | **16 s**（冷缓存复跑，清掉 `/tmp/d2_legs_*`）/ 首轮 21 s | 实测 rc=0 |
| 卡上数值段（每趟 2 腿 × torch_npu 初始化 + 执行 + npz 落盘） | **无实测数据**；按 60–120 s/趟 预留 | 本窗未开卡，估的是 runtime 初始化量级 |
| 变体组加跑九件门（`--combos A,B,C --with-compile-all`，3 组 × (九件 + 2 腿)） | **28 s**（冷缓存） | 实测 rc=0，三组各 `PASS=9` |
| **单窗总预算** | 四趟 ≤ **10 min** + 复核/重跑余量 20 min ⇒ **≤ 30 min** | 估计；建议 30 min 拿不到四趟判决就停窗，不带疑问续跑 |
| 离线判别力矩阵（`D2_probe_matrix.py`，9 TU × 7 组 = 63 次 bisheng） | **23 s**（首轮 ≈35 s），不需卡 | 实测 rc=0 |

省卡要点：**B 趟可以在预算紧张时最后跑甚至不跑**——B 在本窗两腿是空开关（§1.1 静态坐实 + §6.2 指纹
实测 SAME），它的运行时价值只是"阴性对照"；若只能跑三趟，跑 `default / +A / +C`。

---

## 5. 预期结果真值表（红绿组合 → 根因归因）

记 `G`=绿（数值判据通过），`R`=红（编译/运行错或数值不过），`≡`=与基线**同色同数值**，
`R≠`=红且**错值与基线不同**（错误码原文或 rel/dir_disc 数值有差异）。

**读表前必须钉死的三条前置**（读反会得出相反结论）：

1. **有效载荷只有三趟**：`基线 / +A / +C`。`+B` 在本窗两腿是**空开关**（离线证 `so=SAME`，
   见 §6.2），不进真值表，只作**阴性对照**（它若变色 = 判据失效警报，见本节末）。
2. **dw 腿只对 +A 有反应**（§1.1 覆盖关系：dw 读侧只发无 sid 槽的 `_transpose` 件）。
   ⇒ dw 腿在 `+C`/`+B` 与基线同色是**预期**，既不是"读侧被排除"的证据，也不是"变体失效"的证据。
   **读侧归因一律只看 `gemm_l1_intrinsic` 腿。**
3. **先读判决行 3（`so=DIFF`）再读红绿**：某趟 `so=SAME`（开关没进命令行）⇒ 该趟无论红绿
   都**不得**参与归因。

### 5.1 判读阶梯（按顺序停在哪一级就用哪一级的结论）

- **第 1 级 · E-1/裁决②（写侧 ND vs NZ）**：看 `+A` 在 `gemm_l1_intrinsic` 腿的色/值。
  - 变绿 → 现态 ND2NZ 参数取错、朴素 ND 反而对 ⇒ **否决性结果**，P1 需回退，立即停手回报（X1）。
  - 红但错值变（`R≠`）→ 写侧**有影响但非唯一** → 进第 2 级。
  - 红且错值逐值同基线（`≡`）→ 写侧对本载荷**无影响**，E-1 结案为"非主因" → 直接跳第 3 级。
- **第 2 级 · E-2/E-4（读侧 U1 块线性序 / U2 L0 目的步进 / U3 dstGap）**：看 `+C` 同腿。
  绿 / `R≠` / `≡` 三态分别对应 X2、X3、X4。
- **第 3 级 · 交叉验证 `+A+C`**（预算允许时加跑，`COMBO_FLAGS` 已内置该组）：
  把"两处都有份"与"叠加致命"分开（X3 的两个分支）。
- **兜底级 · 三开关全 `≡` 而基线仍红（X5）**：病灶在 §12 三开关覆盖之外的槽位
  （`_transpose` 件 U5、`copy_matrix_cc_to_gm`/l0c2gm 步距 V3、模板几何与 `asc_mmad` 参数）
  ⇒ 需要 `VARIANT-D`（见 §8.4），本窗三开关到此为止。

### 5.2 交叉表（gemm_l1_intrinsic 腿；基线绿的两行是"对照有效性检查"）

| # | 基线 | +A | +C | 归因结论（写回 `X_FRACTAL_EVIDENCE` 的编号） | 下一步 |
|---|---|---|---|---|---|
| X1 | R | **G** | R | **写侧 ND2NZ 参数族取错**（`dstNzC0Stride`/`gCol`/mode），朴素 ND 线性拷贝才是该载荷的正确喂法 ⇒ 裁决② 反转、U1 主嫌改写 | 停手回报（P1 方向问题，非参数微调） |
| X2 | R | **G** | **G** | 写侧与读侧**各有一处真错**（单独回退任一侧都能绿）⇒ 基线红 = 两处错误的叠加，P1 与 P2 都需修订 | 跑 `+A+C`：若绿 ⇒ 确认两错独立；再逐侧单独保留做二分定位 |
| X3 | R | R≠ | R≠ | 两侧**都影响结果**，但都不是充分原因 ⇒ 三槽位之外至少还有一个病灶 | 跑 `+A+C`：绿 ⇒ 两处叠加（回到 X2 语义）；仍红且第三个错值 ⇒ 转 U5/V3（X5 路线） |
| X4 | R | R≡（错值逐值同基线） | R≠ | **E-1 结案为"非主因"**（写侧对该红毫无影响），主嫌锁定读侧 U2/U3 ⇒ E-2/E-4 命中 | 按 `+C` 的错值方向定 U2/U3：折指针起点 `kStart*srcStride+mStart` 与 `dstGap≡0` 两处取值 |
| X5 | R | R≡ | R≡ | **三开关覆盖的槽位全部无关**：E-1/E-2/E-4 一并排除 ⇒ 病灶在 §12 之外（`_transpose` 件 U5、l0c2gm/V3、mad 几何） | 停三开关路线，申请 `VARIANT-D`（§8.4）；别再用本三开关排队耗卡 |
| X6 | **G** | R | **G** | 现态绿：写侧 NZ 化是**必要条件**（裁决② **坐实**）；但 `+C` 与基线同绿 ⇒ **本载荷不敏感**，U1/U2/U3 拿不到区分度（不是"读侧对"的证明） | E-2/E-4 需换 `mStep>1`/`kStep>1` 的更大 tile 载荷才有判据 |
| X7 | **G** | R | R | 现态绿，且 A、C 各自都能打破 ⇒ **两侧都是必要条件**，最强结案（P1+P2 均保留） | 归档，无需续窗 |

### 5.3 意外行（出现即优先归因它，当趟其它结论全部作废）

| 意外 | 含义 | 处置 |
|---|---|---|
| `+B` 与基线**不同色**（红绿或错值变） | sid 槽在真发射里拿到了非零值 ⇒ 静态"mStart 恒 0"模型错，E-3 **复活** | 先查 `mStart` 来源（codegen 直发路？），把 §1.1 覆盖表推翻重做；当趟 A/C 结论不采信 |
| dw 腿在 `+C` 下有反应 | `_transpose` 件被谁改过（Z 未改该件的前提失效） | 停手回报——覆盖关系是全部判读的地基 |
| 基线 G 且 `+A` 也 G 且 `+C` 也 G | 该核根本不依赖 §12 读写侧，或数值判据太松（rel 恒过） | 判据失效警报：先修判据（§8.2 的"错值指纹"），不要按 X6/X7 记结论 |
| 任一趟 `so=SAME`（§3 判决行 3） | 注入链断（头文件没铺对 / monkeypatch 没生效） | 修注入再重跑该趟，**不在窗内改 §12** |

**E-3 的处置（本窗零卡结案）**：`+B` 在两腿的 `.o`/`.so`/`.aibin` 指纹与基线 **SAME**（离线已证），
"开关不改变任何码" ⇒ 卡上不可能给出区分度 ⇒ 对这两条腿 E-3 直接结案为
**否：sid 不是本窗 507015 的触发者**（三重静态证据见 §1.1 与头文件 VARIANT-B 注释块）。
若 `mStart` 将来不再恒 0（codegen 直发 `tl.ascend_load_cbuf_to_ca` 且 m_start 为运行期表达式），
§12 的 B 开关**无需改动即刻可用**（矩阵 M3/M5 已证它在 `mStart≠0` 时 DIFF）。

**"红-红"比对口径约束**：`R≡` 与 `R≠` 的判定必须落在可比数字上（错误码原文 + `rel=`/`dir_disc=`）。
若窗内 `run_numerics` 未落张量级误差（见 §8.2），则 `R≡/R≠` 退化为"错误码原文是否相同"，
X3/X4/X5 的区分度随之下降，**必须在结论里注明该降级**，不得把"同错误码"写成"同错值"。

---

## 6. 离线自证（本次已跑完，可复跑）

> 复跑时间戳：2026-10-11 **18:17**（容器内 10:17 UTC），头树 md5 `ca023c2e…`（三处一致），
> 下列数字全部是**该头树 + 冷缓存**下重新跑出来的（日志 `/tmp/d2_matrix_out.txt`、
> `/tmp/d2_legs_out.txt`、`/tmp/d2_legs_vargate.txt`）。

复跑命令（容器 `cann910b-k`，**不需要卡**）：

```bash
docker cp attempts/D2_probe_matrix.py cann910b-k:/tmp/ && \
docker cp attempts/D2_probe_legs.py   cann910b-k:/tmp/
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;\
  python3 /tmp/D2_probe_matrix.py > /tmp/d2_matrix_out.txt 2>&1;\
  python3 /tmp/D2_probe_legs.py --combos default,A,B,C --baseline /tmp/d2_cache_base \
          > /tmp/d2_legs_out.txt 2>&1'
```

### 6.1 默认路径逐字节等价（三层证据，全 PASS）

| 层 | 判据 | 实测 |
|---|---|---|
| **codegen 面**（最强） | 同一 TU 源码、同一落盘路径，只换头树（pristine `f56432b0` vs D2 落地 `ca023c2e`），整只 `.o` sha256 比 | **9/9 TU IDENTICAL**（含 L1/L2 两枚**真生成码原文** TU：`de37206642a791e3/3024B`、`63cbebca921ddd60/3040B`） |
| **产物面** | 真跑一遍，逐核 `device_kernel.asc` + `executable.so` 指纹对 A2 基线 | **11 核全 `asc=SAME so=SAME`；aibin 集合交集 11/11** |
| **回归门** | `compile_all_asc.py --target ascend` | **9/9 PASS**，`接口腿 PASS=9 FAIL=0 门禁失败数=0`，九件尺寸 `2850/2250/3625/2106/2911/4003/1864/922/2842` 与 A2 基线**逐字一致** |

静态侧：`diff` 改头前后，**删除行只有 3 行**，且都是 `(uint8_t)0, false, …` 被换成
`TL910B_L1_SID(mStart), false, …`（宏默认展开即 `(uint8_t)0`）⇒ 默认路径不是"逐字未改"而是
**"展开后逐 token 等价"**，其正确性由上面三层实证背书，不靠"宏隔离所以安全"的口头断言。

### 6.2 变体确实改变生成码（判别力）

| 组 | 编译判决（真生成码） | 机器码面 | 结论 |
|---|---|---|---|
| `+A` | 9/9 rc=0（两腿真载荷都编过） | `main_kernel` so=**DIFF**、`dW_l0tr_kernel` so=**DIFF**；aibin 交集 0/11 | **活开关** |
| `+B` | 9/9 rc=0 | 两腿 so=**SAME**、aibin 交集 2/11（= 该组两个 aibin 全等基线）| **本窗空开关**（= E-3 零卡结案）；矩阵里 M3/M5（`mStart=1`）DIFF ⇒ 开关本身有效 |
| `+C` | 9/9 rc=0 | `main_kernel` so=**DIFF**（.o 3024→**2768B**，拆趟循环退化成单条指令）、`dW_l0tr` so=SAME（覆盖关系）；aibin 交集 1/11 | **活开关**（仅 gemm_l1 腿） |
| `+B+SID16` | 5/9 rc=0（4 枚 rc=**1**） | `error: the range of 7th parameter must be [0, 15]` @ `port910b_compat.h:1054:21 / 1089:53` | 常量 sid 被 **sema 编译期拦**；这是 §2.4 RT3"sema 只查常量不查运行期值"的**旁证**，但作为真机载荷**编不过 ⇒ 不可用**（已放弃，文档 §5 用的是普通 B 支） |

`aibin 集合交集 k/11` 的分母恒为**基线的 11 个 aibin**（变体组只跑 2 条 §12 腿 ⇒ 分子上限 2）。

尺寸口径补充（免误读）：A 组的 `executable.so` **尺寸不变**（`main_kernel` 恒 25192B）但 sha256 变（default `1250079b185df42c` → A `c6b14ada326557f8` → C `8a3a7191a18eb655`）——ND→NZ 两族指令字节数接近，**判据要用 sha256 而不是文件大小**；C 组在矩阵的 `.o` 面才有尺寸差（真载荷 `L1` 3024B → 2768B，因为 M 向拆趟循环退化成单条指令）。

矩阵全表（`SAME`=该 TU 不吃这组开关，是覆盖关系证据）：

```
   TU                        -D A   -D B   -D B+SID16  -D C   -D A+C  -D A+B+C
   L1_real_gemm_l1_intrinsic DIFF   SAME   RC1!        DIFF   DIFF    DIFF
   L2_real_dw_intrinsic      DIFF   SAME   SAME        SAME   DIFF    DIFF
   M1_gm2l1_bf16             DIFF   SAME   SAME        SAME   DIFF    DIFF
   M2_gm2l1_gcol_fallback    DIFF   SAME   SAME        SAME   DIFF    DIFF
   M3_l12l0a_msplit          SAME   DIFF   RC1!        DIFF   DIFF    DIFF
   M4_l12l0a_kstep1          SAME   SAME   RC1!        DIFF   DIFF    DIFF
   M5_l12l0b_kmajor          SAME   DIFF   RC1!        DIFF   DIFF    DIFF
   M6_dn2nz_delegate         DIFF   SAME   SAME        SAME   DIFF    DIFF
   M7_l12l0a_transpose       SAME   SAME   SAME        SAME   SAME    SAME   ← 覆盖关系实证（Z 未改该件）
```

---

### 6.3 开变体不污染产品面（`--combos A,B,C --with-compile-all`，实测）

三组各自跑 `compile_all_asc.py --target ascend`：**每组都 `接口腿 PASS=9 FAIL=0 门禁失败数=0`**，
九件尺寸 `2850/2250/3625/2106/2911/4003/1864/922/2842` 在三组下**与 default 逐字相同**
⇒ 与 `A3_PREFLIGHT §6` 的结论一致（产品标量路只发 `asc_copy_gm2ub_align`，与 §12 **不同生成码族**）：
**三变体只影响 `*_intrinsic` 对照腿**，窗内因此不必为产品腿重复排队。

## 7. pristine 参照树建法（`D2_probe_matrix.py --ref` 用）

```bash
docker exec cann910b-k bash -c 'P=/usr/local/lib/python3.10/dist-packages/tilelang/src;\
  mkdir -p /tmp/d2_ref_src/src && cp -r $P/tl_templates $P/c_api /tmp/d2_ref_src/src/ &&\
  cd /tmp/d2_ref_src/src && git -C /tilelang show HEAD:src/tl_templates/ascend/port910b_compat.h \
      > tl_templates/ascend/port910b_compat.h'   # 或直接用改头前快照 md5 f56432b0
md5sum /tmp/d2_ref_src/src/tl_templates/ascend/port910b_compat.h   # 期望 f56432b0423174216da50090a1125a24
```
⚠ `src/` 这一层**必须有**，且要带上 `c_api/`（`ascend/debug.h` 要 `#include "c_api/asc_simd.h"`）——
D2 首跑就是栽在这两处（① 节被判 SKIP、② 节 fatal error）。
本机留档：改头前快照 `tilelang/tmp/port910b_compat.h.pre_d2`（`tmp/` 在 .gitignore 内，非仓内产物）。

---

## 8. 需编排者落地（D2 无权/不能自己做）

1. **卡侧头文件铺设**：开卡前把 md5 `ca023c2e01746ca52c87f36bcea93ee9` 的 `port910b_compat.h`
   装进**卡上 tilelang 安装树**的 `src/tl_templates/ascend/`（本窗禁改 `card_run.sh`，无法自动同步）。
   没这步 ⇒ 三开关在卡上不存在 ⇒ 四趟全同色 = 假绿。**窗内第 0 步自检命令见 §3 第一段**，必须跑。
2. **数值判据的采集口径**：`D2_probe_legs.py --card` 用 `run_numerics.py` 现成判决行（`rel=` /
   `dir_disc=`）。X3/X4 需要"**错值指纹**"（红-红之间比对实际张量），而 `run_numerics` 只在绿时打印
   rel——若要落 npz/最大误差，需 B 域窗内在 `run_numerics` 里加一个 dump（本窗禁改该件）。
   预算紧张时可退化为"只看错误码原文（507015 vs 其它）"，X3/X4 的区分度会下降，需在结论里注明。
3. **B 的真机判别载荷缺失**：要让 E-3 在卡上可判，需一条 `m_start` 为运行期非零表达式的
   L1→L0 直发核（现仓内 11 件生成码里没有）。这要 codegen/DSL 面动（`*_kernel.py` 禁改）——
   列为**下窗或 C 域**需求，D2 已在 §12 留好即刻可用的开关。
4. **`_transpose` 件（U5）的第四变体**：dw 腿的读侧只走该件，本窗三开关都碰不到它。若希望 dw 腿
   也能做读侧判别，需要新 `VARIANT-D`（改 `asc_copy_l12l0a/b_transpose` 的双 cfg 字族参数）——
   那需要动 Z 未改的件体，属**提案扩张**，未擅自落地。
5. **A2 基线缓存目录的卡侧对应物**：`--baseline` 需要一个"已知good 的产物目录"。容器里是
   `/tmp/d2_cache_base`（离线，A2 时代留档）。卡侧若无对应留档，第一趟 `default` 即成为后续三趟的基线
   （此时"与基线等价"改由 §6.1 的离线三层证据保证，窗内只需 A/C 的 `so=DIFF` 出现）。

---

## 9. 本窗产物清单与禁改遵循

新增：`attempts/D2_variants.md`（本文件）、`attempts/D2_probe_matrix.py`、`attempts/D2_probe_legs.py`
修改：`tilelang/src/tl_templates/ascend/port910b_compat.h`（仅 §12 区，+66 行/-3 行）
同步：`release/ascend/port910b/patches/port910b_compat.h`（md5 与 trunk 一致）
未碰：`train_step.py`、`autograd_asc.py`、`card_run.sh`、`patch_bisheng.py`、`compile_all_asc.py`、
`run_numerics.py`、`ascend_env.py`、任何 `*_asc.py`/`*_kernel.py`、任何 `tasks.md`；全程未用 git 写操作。
