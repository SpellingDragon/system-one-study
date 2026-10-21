# R6_DBETA_SINGLE_FAULT.md — P1-1zk：fork GDN bwd `dbeta` 单点病灶离线取证（静态审 + 考古 + 上卡腿配方）

> 执行代理：**R6 取证**（openspec-apply 战位）｜变更：`teacher-p2-production-full / p2-13-ascend-runtime`｜任务：**P1-1zk**
> **纯分析波**：不修代码、不上卡、禁 git、不动 runs 历史；唯一落盘产物即本件。
> 分析对象：`release/ascend/kernels/gdn_asc.py`（md5 `676c07936547e9d897d3b2366a0317e0`，✅ 本波实测复核）及其 `gdn_bwd_asc_impl`。
> 三态标注：**✅ 实测/引窗判决原文**（本波直接读到的文件与原始日志）｜**⚠️ 引用**（历史台账/他代理记录，本波未复跑）｜**❓ 推断**（未经判决实验）。
>
> **一句话结论**：五个 PASS 输出已经把 `dbeta` 表达式的**全部输入值**证真（`du` 由 `dv` PASS 证真、`p` 与 `v_scr[j]` 由 `dq/dk/dh0` PASS 经 `uj` 链证真，`bt` 恒非零），病灶收窄为**仅剩三个分岔点**——(i) `:425` 对 `(v−p)` 的二次求值错位、(ii) `db_acc` 循环携带累加被错算、(iii) `DBeta[h,t]` 的秩-2 GM store 错——**输入正确而输出错的"单表达式确定性错算"假说由此获得结构性支撑**；上卡腿按此三分岔设计判据（§3），一枪可判活/判负。

---

## 0. 背景锚点（本波引用的已定案事实）

- 窗十一旗真判决（⚠️ 引用 `runs/1013-p2-13-oncard-window10-align-workaround/notes.md` §[4]）：
  ```
  dq 6.999e-08 PASS / dk 9.322e-08 PASS / dv 6.629e-08 PASS
  dbeta 8.002e-01 FAIL（与无旗态逐字同值）
  dg 1.016e-07 PASS / dh0 6.176e-08 PASS
  ```
  `--cce-auto-sync` 不救 dbeta ⇒ "fork 缺核内 vector↔scalar 同步"（窗八附最强候选）**证伪**。
  注：该六项值目前**只存在于 notes.md**，artifacts 里 `w10_flag.log` 只有第一枪的 `libhccl.so` 失败（✅ 本波实查）——数字溯源缺口见 §4 风险 R-3。
- 窗八原始日志（✅ 本波实读 `runs/1012-p2-13-oncard-window8-a2/artifacts/w8_a2.log:23-29`）：
  ```
  ITEM dq rel= 6.999e-08 verdict= PASS
  ITEM dk rel= 9.322e-08 verdict= PASS
  ITEM dv rel= 6.629e-08 verdict= PASS
  ITEM dbeta rel= 8.002e-01 verdict= FAIL
  ITEM dg rel= 1.016e-07 verdict= PASS
  ITEM dh0 rel= 6.176e-08 verdict= PASS
  ITEM adjoint rel= 5.999e-10 verdict= PASS
  NUM-delta_bwd FAIL first=dbeta rel=8.002e-01 absdiff=7.342e+00 thresh=2e-4
  ```
  同窗 `w8_d3.log` 顺序探针（✅）：`none==dirty==zero==e609a6ab`，`ORDER-VERDICT 逐位同 ⇒ 顺序无关(H-b 证伪)`。
- 判据语义（✅ `release/ascend/port910b/run_delta_card.py`）：`rel = ||meas−ref||₂ / ||ref||₂`（:66-69）；`absdiff = max|meas−ref|`（:289）；逐项 `ITEM` 行容错不早退、非有限判 FAIL（:277-309）；判形 `H,TT,DK,DV,CORES = 2,8,8,8,1`（:49），`BETA = U[0.25,0.75)`（:78）⇒ **`bt` 恒非零**（下文排除链的关键前提）。
- 三修史已全部真机证伪（⚠️ 引用窗七/窗八台账），**均属"UB 未初始化"假说时代，P1-1zg 线已关，本波不重开**：M4 入口清零（`M4_UBPREZERO.md` 头部自注"证伪并已回退"）、A2 预热（窗八 a 轮）、v2 头文件预清零（窗八结论 3 归并）。窗八结论 2 原文："`dbeta 8.002e-01` 跨 3 个 boot、4 个窗口、两种变体逐字复现 ⇒ 这不是随机未初始化读，是**确定性错误计算**"。
- 变体事实（⚠️ 引用 `K4_RESULT.md` E1–E3）：判决腿与生产腿设备上都走 `ub`（键 `c2ubz`）；但窗四 V4 记录 **device `pure` 同值 FAIL**（`w4_d3_pure.log`，✅ 本波实读：`ITEM dbeta rel= 8.002e-01 verdict= FAIL`、三跑指纹恒 `e609a6ab`）⇒ 错值与 UB 槽内容无关，是 ub/pure **共享标量数学/公共 codegen** 的确定性错算（D3_NEXTSHOT 合并判词表"任意 | ub FAIL / pure FAIL"格）。

---

## 1. dbeta 全链路静态审（`gdn_bwd_asc_impl`，逐语句 + 内存层级/访存形态）

审查范围：`gdn_asc.py:286-491`。`dbeta` 计算链只在 `ub` 档 `:402/:425/:426` 与 `pure` 档 `:463/:481` 各三处（✅ 本波逐行读）。以下为 **ub 档**（设备真跑档，`run_delta_card.py:60-63` `_variant()` ascend→ub）逐语句：

| # | 源语句（gdn_asc.py:行号） | 内存层级 / 访存形态 |
|---|---|---|
| 1 | `:401` `bt = Beta[h, t]` | GM→标量寄存器（fp32，动态 `t` 索引） |
| 2 | `:402` `db_acc = T.float32(0.0)` | 寄存器清零，**每 token 一次**（循环携带累加器的起点） |
| 3 | `:405-408` `dec_ub[i]=T.exp(G[h,t,i])` / `k_ub[i]` / `q_ub[i]` | GM→UB 标量逐元素（exp 走 codegen expf） |
| 4 | `:409-411` `v_scr[j]=V[h,t,j]` / `do_scr[j]=DO[h,t,j]` | GM→UB 标量逐元素 |
| 5 | `:415-419` `p`、`du` 对 `i` 归约 | UB（dec/k/q/lam/do）+ GM（`Hist[h,t,j,i]` 四维索引，`:418`）→寄存器累加 |
| 6 | `:420` `uj = bt * (v_scr[j] - p)` | **`(v−p)` 首次求值**：UB 读 `v_scr[j]`（运行期偏移 j）+ 寄存器 `p`；结果进 `uj` |
| 7 | `:421` `dp = T.float32(0.0) - bt * du` | 寄存器 |
| 8 | `:422-423` `u_scr[j]=uj` / `dp_scr[j]=dp` | 寄存器→UB（运行期偏移 j；**发生在 #6 之后**） |
| 9 | `:424` `DV_[h, t, j] = bt * du` | 寄存器→GM 标量 store（秩-3 索引；dv 输出，**PASS**） |
| 10 | `:425` `db_acc = db_acc + du * (v_scr[j] - p)` | **`(v−p)` 二次求值**：再次 UB 读 `v_scr[j]`（#8 的 UB 写、#9 的 GM store 之后）+ 复用寄存器 `p`；结果累加进 `db_acc` |
| 11 | `:426` `DBeta[h, t] = db_acc` | 寄存器→GM 标量 store，**秩-2 索引**（`h*seq+t`，`seq` 为动态符号），j 循环之外、每 token 一次 |
| 12 | （趟 2）`:429-445` `dq_acc/dk_acc/da_acc` 各自对 `j` 归约后 `:443-445` 单存 | UB（lam/u/dp/dec）+ GM（Hist）→寄存器累加→GM 标量 store（秩-3 索引） |
| 13 | （出口）`:484-487` `DH0[h,j,i]=lam_ub[j,i]` | UB→GM 标量直搬（无归约、无 seq 索引） |

`pure` 档同构（✅）：`db_acc` 累加在 `:463`（`du * (vj - p)`，`vj` 于 `:450` 直读 GM）、落盘 `:481`；每 token 清零 `:402` 两档共用。

### 1.1 形态对照表：dbeta vs 五 PASS 输出（同 kernel、同编译、同输入）

| 输出 | 源语句 | dst buffer（形状/秩） | dtype | reduce 形态 | store 路径 | 真机 |
|---|---|---|---|---|---|---|
| dq | `:443` | `DQ (h,seq,dk)` 秩-3 | fp32 | `dq_acc` 寄存器累加：**i 外层 × j 内层**，每 (h,t,i) 一存 | 标量 GM store，`write_gm_bypass_dcache`（⚠️ 生成码 L72 同族），地址 `(h*seq+t)*dk+i` | PASS 6.999e-08 |
| dk | `:444` | `DKG (h,seq,dk)` 秩-3 | fp32 | `dk_acc` 同上 | 同上 | PASS 9.322e-08 |
| dv | `:424` | `DV_ (h,seq,dv)` 秩-3 | fp32 | **无归约**：`bt*du` 逐元素直存（j 循环内） | 标量 GM store（⚠️ 生成码 L72），地址 `(h*seq+t)*dv+j` | PASS 6.629e-08 |
| dg | `:445` | `DG (h,seq,dk)` 秩-3 | fp32 | `da_acc` j 归约后 × `dec_ub[i]`（UB 读） | 同 dq | PASS 1.016e-07 |
| dh0 | `:487` | `DH0 (h,dv,dk)` 秩-3 | fp32 | 无归约：UB→GM 直搬 | 标量 GM store，地址 `(h*dv+j)*dk+i`（**无 seq**） | PASS 6.176e-08 |
| **dbeta** | `:426` | **`DBeta (h,seq)` 秩-2** | fp32 | `db_acc` 跨**整段 dv** 循环携带（`:402` 每 token 清零 → `:425` 累加），每 (h,t) 一存 | 标量 GM store（⚠️ 生成码 L75），地址 `h*seq+t` | **FAIL 8.002e-01** |

**形态差异只剩四条**（✅ 由上表直接得出）：
1. **秩-2 dst**：`DBeta` 是唯一无通道维的输出（索引 `h*seq+t`，动态 `seq` 只进 stride 一次）——但 `DV_` 的 stride 同样含动态 `seq` 且 PASS ⇒ "动态 stride 寻址"本身不致错，只剩"秩-2 地址表达式"这一窄面。
2. **整段携带累加**：`db_acc` 是唯一跨**整个** `dv` 循环携带的累加器；`dq_acc/dk_acc/da_acc` 是"每个 i 一段"（`:430-432` 逐段清零）。
3. **`(v−p)` 二次求值**（`:420` 与 `:425`）：同一 j 迭代内同一表达式求值两次，中间隔着 #8（两次 UB 写）与 #9（GM store）——**这是全函数唯一一处"同表达式重算"**，且是 dbeta 专属（`uj` 走 `:420` 的那次求值，喂给 PASS 的 dq/dk/dh0）。
4. **`v_scr[j]` 的二读形态**：`v_scr[j]` 全函数被读两次（`:420`/`:425`），都在同迭代内；PASS 输出对 UB 运行期偏移读都只读一次（`u_scr[j]`/`dp_scr[j]`/`do_scr[j]` 各一次）。

生成码侧佐证（⚠️ 引用 `D3_DBETA_EVIDENCE.md §2.3`，容器重编 8×8 `md5 8b11e54e…`，**非 C 段真跑件**，见 §4 风险 R-1）：
```
51:    db_acc = float(0x0p+0f);                                   ← 每 token 清零在
68:    uj = (bt * (((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 104)] - p));   ← (v−p) 一求
72:    tl::write_gm_bypass_dcache(DV_ + …, (bt * du));
73:    db_acc = (db_acc + (du * (((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 104)] - p)));  ← (v−p) 二求
75:    tl::write_gm_bypass_dcache(DBeta + …, db_acc);
```
即生成码层 `(v−p)` **确实重复求值**（`p` 为寄存器复用、`v_scr` 两次 UB load）；`DBeta` 与 `DV_` 共用同一 store 原语 `write_gm_bypass_dcache` ⇒ store 原语无差异，地址表达式是唯一分岔。

### 1.2 排除链（本波静态审的最大产出）：dbeta 的输入值全部已被 PASS 输出证真

1. **`du` 正确**：`dv = bt*du`（`:424`）真机 PASS ⇒ 每 j 的 `du` 值正确（`bt` 是标量乘子，`BETA∈[0.25,0.75)` 恒非零，✅ `run_delta_card.py:78`）。同理 `dp = −bt*du`（`:421`）正确。
2. **`p` 与 `v_scr[j]`（`:420` 处的求值）正确**：`uj = bt*(v_scr[j]−p)`（`:420`）→ `u_scr[j]`（`:422`）→ 趟 2 的 `dq_acc/dk_acc`（`:438/:439`）与 `Λ` 递推（`:440-442`）→ `dq/dk/dh0` 全 PASS ⇒ `uj` 正确 ⇒ 因 `bt≠0`，`(v_scr[j]−p)` **在 `:420` 的那次求值**结果正确。
3. **`Hist/dec/k/q/do` 读取面正确**：它们是 `p/du/Λ` 的输入，由 1/2 传递证真；`dg` PASS 另证 `Hist*dst*a_i` 链（`:436-445`）。
4. ⇒ **dbeta 表达式 `du*(v−p)` 的三个操作数在各自"首次使用点"都是正确值**（M4 §1.4 的"数学悖论"由此形式化，⚠️ 引用）。**在"输入正确、代数正确（CPU 同式 PASS，`D3_DBETA_EVIDENCE.md §3.2` 四方同构）"前提下，输出仍确定性错 ⇒ 病灶必落在"算/存"的三个分岔点之一**：
   - **(i)** `:425` 的二次求值拿到 ≠ `:420` 的值（寄存器复用/UB 重读错位）；
   - **(ii)** `db_acc` 的循环携带累加（含 `:402` 每 token 清零）被 codegen/执行错算；
   - **(iii)** `:426` 的 `DBeta[h,t]` store 错（值被丢/写错址）。
   三者之外无路（❓：排除链依赖 `dq/dk/dh0` 的 PASS 确实覆盖 `u_scr/dp_scr/Λ` 路径——这在源码上成立，见 :438-442）。

---

## 2. 候选机理排序（每条附可证伪预测）

**先验总注**：窗八已定"确定性错误计算"（✅）、窗十一旗已排除同步机理（⚠️）、M4/L4b 已证生成码静态审计对 read-first 类**无判别力**（⚠️ `L4B_RESULT.md §0B`："已知坏件 gdn_bwd 生成码也无任何 read-first ⇒ 静态审计对本缺陷类无判别力"）——故本节排序依据是**形态唯一性 × 排除链剩余面**，不是静态"看图说话"；每条都给了上卡一枪可判的预测。

### 候选 A（最优）：`:425` 对 `(v−p)` 的二次求值错位（寄存器复用 / UB 重读）——❓ 推断，先验 ≈ 40%
- **机理**：`du * (v_scr[j] - p)` 中的 `(v_scr[j]−p)` 与 `:420` 是同一表达式的两次独立求值，中间夹着 `:422-423` 的 UB 写与 `:424` 的 GM store。若 codegen 在此交错下让第二次求值读到错的 `p`（寄存器被 :422-424 的地址计算/写路径复用污染）或错的 `v_scr[j]`（UB 重读错位/错 lane），则 `uj`、`dv` 全对而 dbeta 错——**与"唯 dbeta 单点错"完美自洽**，也是形态对照中唯一"dbeta 专属且未被 PASS 输出覆盖"的读形态（§1.1 差异 3/4）。
- **可证伪预测**：把 `:420/:425` 的 `(v−p)` 提为**单点求值**（`diff = v_scr[j] - p; uj = bt*diff; …; db_acc += du*diff`，纯机械等价改写）后上卡：
  - 若 dbeta `8.002e-01 → <2e-4` ⇒ **判活坐实**（且 fork 路线可救，P1-1zh 可提前收）；
  - 若不变（逐字 `8.002e-01`）⇒ A 证伪，转 B/C。
  - 附带无改预测：卡上 `DBeta` 原始值若与"用 `:420` 的 `(v−p)` 正确值"算出的和不符、却与某个**错位重读**（如 `do_scr[j]`/`u_scr[j]`/`dp_scr[j]` 顶替 `v_scr[j]`，偏移 112/72/80 顶 104，⚠️ 槽位表引 D3 §2.3）相符 ⇒ 直接指认错位对象。

### 候选 B：`db_acc` 循环携带累加/清零被错算——❓ 推断，先验 ≈ 25%
- **机理**：`db_acc` 是唯一跨整段 `dv` 携带、每 token 一存的累加器（§1.1 差异 2）。若 codegen 把携带链拆解/只累计部分 j/丢掉 `:402` 每 token 清零（跨 token 泄漏），则确定性错。
- **可证伪预测**（全部可用卡上 meas 在 host 离线判）：
  - 若 `:402` 清零丢失（跨 token 泄漏，`tt` 逆序遍历 t）⇒ `meas[t] = Σ_{t'≥t} ref[t']`，且 `meas[TT-1] == ref[TT-1]`；
  - 若只累计前 m 个 j ⇒ `meas == Σ_{j≤m} du_j(v_j−p_j)`（m=0..dv-1 共 8 个候选和逐一对拍）；
  - 若累加变"只留末项/首项" ⇒ meas == 单项 `du_m(v_m−p_m)`。
  - 任一命中 ⇒ B 坐实；全不命中 ⇒ B 证伪。
  - 改写臂预测：把归约改成"逐 j 写 `db_scr[j]`（UB 或 GM）再显式求和"或双变量轮换累加 ⇒ 转对即判活。

### 候选 C：`:426` `DBeta[h,t]` 秩-2 GM store 错址/丢写——❓ 推断，先验 ≈ 10%（低）
- **机理**：唯一秩-2 输出，地址 `h*seq+t` 动态 stride 单点。若 store 落错址，`DBeta` 区读回未写值。
- **反证先验**：`rel=8.002e-01 / absdiff=7.342` 跨 boot/窗/变体**逐字稳定**（✅ w8 原始日志 + ⚠️ 窗十）——"读回 torch.empty 垃圾"应随 boot 变，故 C 先验压低；但"确定性错址 + 邻区确定值读回"仍可稳定，不能归零。
- **可证伪预测**：宿主把 `gb` 输出张量**预填哨兵**（如全 `1e30` 或 bit-pattern `0x7f7fffff`）再发射：
  - 若读回哨兵混合 ⇒ store 错址/丢写坐实（且可从"哪些位置被写"反推真实 stride）；
  - 若读回全为非哨兵的确定值且逐字 `8.002e-01` ⇒ C 证伪。
  - 附带预测：若 C 真，改 `TT=7/9` 重编（缓存键不含 seq，⚠️ `gdn_asc.py:601` 注）错位形态随 seq 变；A/B 真则不随。

### 候选 D：codegen 基础设施层（调度/寄存器分配 fallback）——⚠️ 观察 + ❓ 因果，先验 ≈ 25%
- **观察**（✅ 本波实读 `w8_a2.log`/`w8_d3.log`/`w4_d3_*.log`）：`gdn_bwd_impl` 编译窗口内稳定出现
  `Warning: Python Z3 scheduler failed with exception: Z3 scheduling failed: solver returned unsat, falling back to topological sort`。
  该告警亦见于其他内核日志（w4_gradcheck 等），**与 dbeta 的因果未证**；但它说明 gdnb 的调度走了 fallback 路径——A/B 若在指令调度/寄存器分配层出错，此为上游温床。
- **可证伪预测**：dump 真跑件生成码后比对容器重编件（§3 腿 0）：若真跑件的 `db_acc` 段调度/临时变量与容器件不同（不同编译路径）⇒ 支持 D；若逐字同 ⇒ D 降权。另：给 gdnb 编译强制 `opt_level` 降档/换调度器重编一发，dbeta 随之改变或转对 ⇒ D 坐实。

### 已排除（对照窗九 TA 侧判据矩阵 `window9-ta-bwd/notes.md` 与既有实证，不再进候选）
1. **UB 未对齐访存族（507015，窗九 #13–19 / 窗十 [2][3] 对齐版全 FAIL）**：那是 TA 通道 `LowerParallelToVector`×UB 布局的缺陷域（⚠️ 窗九生成码 diff：Broadcast 展开致 buffer 偏移漂移）；fork gdnb 六项在真机**全部跑出、无芯片异常**（✅ w8_a2.log）⇒ dbeta 病灶不是未对齐访存。
2. **j 广播/Broadcast 展开（窗九 #13/14/15/17/18）**：fork 生成码 **0 DMA、0 Broadcast、纯标量逐元素**（⚠️ `D3_DBETA_EVIDENCE.md §2.4a`）⇒ 结构上不适用。
3. **TA tile API 四纪律面（窗九 b 轮四枪：标量 ctx exp 缺失、Parallel 广播赋值、GM 标量 store 须 `Scope("V")`、UB 运行期偏移标量读）**：fork 在这些形态上均有 PASS 输出同路径实证——`dec_ub[i]=T.exp(...)` 元素级 exp 进 `dg` PASS、`u_scr[j]/do_scr[j]` 运行期偏移 UB 读进 `dq/dk` PASS、`DBeta` 与 `DV_` 同用 `write_gm_bypass_dcache`（⚠️ 生成码 L72/L75）⇒ 这些形态**单独**不致错（但"二读 + 交错写"形态未被 PASS 覆盖，正是候选 A 的立足点）。
4. **同步机理（窗八附最强候选）**：`--cce-auto-sync` 一旗真机证伪（⚠️ 窗十 notes §[4]）。
5. **UB 未初始化/stale 族（三修史）**：M4/A2/v2 全证伪（⚠️ 窗七/窗八）；窗八顺序探针 `none==dirty==zero`（✅）⇒ 与 UB 前态无关。

---

## 3. 上卡腿配方（P1-1zj 验收窗用；本波只写配方，不实施）

**总原则**：一窗三腿（腿 0 无改取证 → 腿 1 无改判活（候选和对拍）→ 腿 2 改写臂），全部同窗同 boot 跑完（历史教训：窗四/窗八的 prefill 现象跨 boot 不可复现，⚠️）；判据脚本按候选逐条可开关；**先容器编译门后上卡**（R20/R-窗九附 1）。

### 3.0 fork 生成码 dump（补 D3 §4.3 盲区：C 段真跑件从未入库）
卡上约定（✅ `card_run.sh:60` 与 `:307-311`）：`TILELANG_CACHE_DIR` 每键一目录，内含 `device_kernel.asc + host_kernel.c + manifest.json + params.json`；窗收尾打 `/tmp/sys1_kernels.tgz`。
```bash
# 卡上（与判决同一进程同一目录，锁"生成码↔判决"对位；勿 mktemp 丢弃）
export TILELANG_CACHE_DIR=/tmp/sys1_r6_dump && mkdir -p $TILELANG_CACHE_DIR
python3 release/ascend/port910b/run_delta_card.py --target ascend --bwd | tee /tmp/r6_bwd.log
# 定位 gdnb 真跑件（键 ascend|gdnb[ascend]|h2k8v8|c2ubz）
for d in $TILELANG_CACHE_DIR/*/; do
  grep -l 'void gdn_bwd_impl_kernel(' "$d/device_kernel.asc" 2>/dev/null && echo "FOUND $d"
done
tar czf /tmp/sys1_r6_kernels.tgz -C $TILELANG_CACHE_DIR .   # 回传入库（对位凭据）
```

### 3.1 diff 判据（五 PASS 输出生成码段 vs dbeta 段）
对 `device_kernel.asc` 逐段抽取并对照（判据脚本开关 `--diff`）：
| 段 | 抽取锚（引 D3 §2.3 行号形态） | 判什么 |
|---|---|---|
| dbeta 段 | `db_acc` 声明/清零/累加行 + `DBeta` store 行（L24/42/51/73/75 形态） | 累加器清零次数与位置（每 token 一次？）、归约循环边界（j=0..dv？）、store 地址表达式 |
| dv 段（PASS 对照） | `DV_` store 行（L72 形态） | 同一 j 循环内 PASS store 的地址/临时变量形态 vs dbeta |
| dq/dk/dg 段（PASS 对照） | `dq_acc/dk_acc/da_acc` 累加 + `DQ/DKG/DG` store | "每 i 一段"累加的清零/携带形态 vs dbeta 的整段携带 |
| 二次求值点 | `uj = (bt*(buf[j+104]-p))` 与 `db_acc=(db_acc+(du*(buf[j+104]-p)))` | **(v−p) 是否被 CSE 合并**；两次 `buf[j+104]` load 之间是否有写 `buf[j+72]/buf[j+80]` 与 GM store；`p` 的寄存器分配是否在两次使用间被复用 |
**命中模式**（❓ 判据预期）：二次求值未合并 + `p`/`v_scr` 临时变量在 L68→L73 间被改写 ⇒ A 坐实；`db_acc` 清零缺失/循环边界错 ⇒ B；store 地址表达式与 `DV_` 段不一致（秩-2 stride 错）⇒ C。

### 3.2 判据脚本草案（python，按候选逐条可开关；只写在此处，本波不落盘）
```python
# r6_dbeta_judge.py（草案）——用法：卡上 dump 出 DBeta meas（.npy）后在 host 跑
# 开关：--arm {sum_correct,cross_token,partial_j,wrong_slot,sentinel,diff_gen}
import numpy as np
H, TT, DK, DV = 2, 8, 8, 8
# 1) host fp64 复算逐 j 中间量（复用 run_delta_card 的输入口径 seed=11 + golden adjoint 同式）
#    du[j], p[j], uj[j] = bt*(v[j]-p[j])  （bt∈[0.25,0.75)，run_delta_card.py:78）
# 2) 候选和族（逐 t）：
S_correct    = [sum(du[j]*(v[j]-p[j]) for j in range(DV)) for t in ...]      # 对拍 ref（应≈ref）
S_cross_tok  = [sum(S_correct[t:]) for t in range(TT)]                        # 候选 B 子预测①
S_partial_j  = [[sum(du[j]*(v[j]-p[j]) for j in range(m+1)) for t ...] for m in range(DV)]  # B②
S_wrong_slot = [sum(du[j]*(X[j]-p[j]) for j ...) for X in (do,u,dp,dec)]     # 候选 A 错位重读
# 3) 判决：meas 与哪个候选逐位/近位（<=1e-6）吻合 ⇒ 指认机理；全不吻合 ⇒ A/B 判负转 C/D
#    另 --sentinel：检查 meas 中哨兵残留模式（C）；--diff_gen：跑 §3.1 的段 diff
```
配套改动臂（**实施留 P1-1zj 授权**，各臂最小化、一臂一判）：
- **A 臂**：`:420/:425` 单点求值（`diff` 变量复用）——预期 dbeta 转对；
- **B 臂**：`db_acc` 改 `db_scr[j]` 落 UB 再显式求和 / 双变量轮换——预期转对；
- **C 臂**：`DBeta` 显式 `T.Scope`/换 store 形态或改秩-3 `(h,t,0)`——预期转对；
- 每臂上卡前过容器编译门（`--no-run` 键仍须 `gdnb[`，假绿守卫沿用 `run_delta_card.py:257-259`）。
**真值表**：A 臂转对 ⇒ 判活（P1-1zk 彩票兑现，fork 训练管线提前全绿、P1-1zh 提前收）；A/B/C 全不转对 ⇒ 判负（fork 路线盖棺，价值＝定案 + issue 取证素材）。

---

## 4. 结论与风险

### 结论
1. **最优候选 = A（`:425` 二次求值错位）**，先验 ≈ 40% ❓。依据：排除链（§1.2）把输入全证真后，A 是形态对照中唯一"dbeta 专属且五 PASS 未覆盖"的分岔；与"确定性单点错算"“ub/pure 同值"（共享标量数学段）三观测量同时自洽。
2. **判活/判负先验**：判活（A 或 B 改写臂一枪转对、fork 复活）≈ **30–40%** ❓；判负（三分岔全空、机理沉到生成码之下如 D 层）≈ **60–70%** ❓。理由如实：有利面＝病灶已收窄到单表达式的算/存三分岔、判据可一枪覆盖、`du/p/v` 输入全对意味着"改写表达式形态"有直接抓手；不利面＝三修史显示"看起来该修的地方改了不动"（M4 生成码 diff 纯新增却改变数值、A2 触发了没修好）、L4b 已证静态审计对本缺陷类无判别力、且 C 段真跑件从未对位（真机生成码可能与容器重编件不同）。
3. **独立定案价值**（即使判负）：①"dbeta 输入全对"排除链把假设空间砍掉整族（UB 态/同步/未对齐/exp/Hist 面），后续任何 fork 复活尝试都须先过此三分岔；②窗口十病灶收窄结论获得静态结构解释；③上卡腿配方（§3）自带生成码对位取证，补掉 D3 §4.3 盲区。

### 风险
- **R-1 生成码对位盲区**：⚠️ D3 的生成码是容器**重编**件（8×8 `md5 8b11e54e`），C 段真跑件用 `mktemp -d` 缓存从未入库（`D3_DBETA_EVIDENCE.md §4.3` 自认"不能 100% 断言逐字节相同"）——所有生成码层结论带此前提；§3.0 的同进程 dump 是解药。
- **R-2 窗十数字只在 notes**：`--cce-auto-sync` 六项判决值无原始日志落 artifacts（✅ 本波实查 `w10_flag.log` 只有第一枪失败）——引用标 ⚠️，复判需重跑（不影响"与无旗态逐字同值"的定性结论，窗八 `w8_a2.log` 已有无旗态 ✅ 原文）。
- **R-3 改写臂的等价性**：A/B/C 臂是机械等价改写，但 M4 先例表明"语义中性"的改动会改变设备数值（⚠️ 窗七）——每臂必须先过 host 逐字不变门（`--target cpu --bwd` 六项 + pytest 340）再上卡，且一次只开一臂。
- **R-4 rel 口径**：`rel` 是 L2 范数比（`run_delta_card.py:66-69`），`8.002e-01` 不是逐元素误差；候选和对拍（§3.2）应以**张量逐位**比对为主判据，rel 仅作聚合参考。

---

## 5. 复跑命令（离线面全部可溯源）
```bash
REL=$REPO_ROOT/release
md5 $REL/ascend/kernels/gdn_asc.py                      # 期望 676c07936547e9d897d3b2366a0317e0
sed -n '399,426p' $REL/ascend/kernels/gdn_asc.py        # dbeta 链（ub 档）逐语句
sed -n '20,32p' $REL/runs/1012-p2-13-oncard-window8-a2/artifacts/w8_a2.log   # 六项 ITEM + absdiff 原文
sed -n '18,32p' $REL/runs/1012-p2-13-oncard-window8-a2/artifacts/w8_d3.log   # 顺序指纹三态原文
grep -n 'dbeta\|8.002' $REL/runs/1012-p2-13-oncard-window4-pin/artifacts/w4_d3_pure.log  # device pure 同值
```

---
*R6 取证件（P1-1zk 离线部分）。三态：✅ 实测/引窗判决原文 / ⚠️ 引用 / ❓ 推断。未实施修复、未上卡、未用 git；产物仅本件。*
