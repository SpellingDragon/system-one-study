# 窗十一台账：B 线 v2 真机首验（驱动层阻断→修复）+ 真形档 + D 线腿0&A 臂

- 窗：`window11_card.sh` + `run_window11_local.sh`（port910b 根）；实例 1017 新 boot（/tmp 全清，ta_pkg 重传 sha `d5607ec6`）
- 判据前置：本机 pytest 门 **340 passed, 6 skipped**（A 臂授权）→ 发射 `W11_A2=1`
- 现场档：`artifacts/w11_all.log`、`w11_artifacts.tgz`、`w11_sys1_r6_kernels.tgz`(68444B)、`w11_ws_cache.tgz`(104971B)

## [1] B 线 v2 L0：CPU 双门全绿，**7 真机 case 全 IndexError**（驱动层阻断，非内核/设备）

- CPU 门：`adjoint 5.999e-10` / `chunk-math 2.807e-15`（31 条 chunk-* 全 PASS，含 tail_1/2/44）
- 7 case（六梯度×各 case no early exit）：全部 `IndexError: list index out of range`——秒级、无编译痕迹 ⇒ **编译成功后**的运行期参数装配错
- [2] 真形档同错（无 try/except，堆栈全出）：
  `gdn_bwd_v2_solve_a(...)` → `kernel.py:201 __call__` → `adapter.py:455 lambda_forward` → `cython_wrapper.pyx:139 tensor = inputs[ins_idx]`

## [2] 病灶定位（诊断腿 1 `w11_diag1.sh`，决定性证据）

- cumsum 对照：`nparams=2 result=[1] ws=[] agm=[]` → 调用 OK
- solve_a：**`nparams=8 result=[7] ws=[] agm=[6,7]`**；param[5]=真 A `[1,1,64,64]` fp16 落入输入消费位
- 机理：CV pass 在 prim_func 声明 6 参之后**追加 2 个 auto_gm 参数**（params=8）；CythonAdapter 将 `out_idx=[-1]` legalize 为 `len(params)-1 = 7` → 指向尾部 agm 槽 → 真输出 `param[5]` 不被 result_idx 捕获 → 5 实参时 `inputs[5]` 越界
- 佐证：6 实参 hack 调用通过（`D1-A6-OK`，但返回 `[1,64,64]` 错形状）——坐实"装配错位"而非其他
- **修复**：solve_a `out_idx=[-1]→[5]`、cumsum `[-1]→[1]`（防御统一；scan1/2a/2b 本就显式索引）
- 语义边界：`-1` 的 legalize 与"编译期追加参数"组合是 TA 适配层面 —— 显式索引对追加免疫（追加均在尾部）

## [3] D 线腿0：无改 ascend 判决复采 + 生成码同进程 dump

- 六项：`dq 6.999e-08 / dk 9.322e-08 / dv 6.629e-08 / dbeta 8.002e-01 FAIL / dg 1.016e-07 / dh0 6.176e-08`；`absdiff=7.342e+00`
- **与窗十无旗态逐字同值** ✓（R-2 盲区修复：原始日志首次落 artifacts）
- dump：`w11_sys1_r6_kernels.tgz 68444B`（gdnb 真跑件与判决同进程同 boot cache，补 D3 §4.3"容器重编件"盲区）

## [4] A 臂一枪（候选 A 单点求值）：**扰动但未转对**

- 本机门：patch :420/:425（`diff = v_scr[j] - p` 提升+复用）→ pytest **340 passed** → 还原 md5 `676c0793` 逐字复原
- 卡上 CPU 门：改后 vs 基线 **IDENTICAL**（c 后端六项逐字同 ⇒ 语义等价已证）
- 卡上 ascend 一枪：`dq/dk/dv/dg/dh0` **五项逐字不动**；**`dbeta 8.002e-01 → 5.997e-01`**（absdiff 7.342→4.435）仍 FAIL
- 判读：改写精确扰动 dbeta 路径 ⇒ 病灶确在 :420-425 段；但单点求值不足 ⇒ "二次求值错位"单机理不成立（或叠加第二机制）；五项不动 ⇒ 改写未破坏其余输出
- 回滚：`REVERTED=0` + POST-MD5 `676c0793` 逐字复原

## [5] 真形档 workspace：被驱动层阻断（修复版重跑中 `w11b`）

- 首跑：`WS-EXIT=1`、cache 2 条目（编译成功即炸于 solve_a 装配）；声明量 307.5 MiB 未获运行时验证
- 重跑：见下节（修复后）

## [6] 真跑件生成码取证（D 线腿 0 dump 离线解剖，`4ca1fd9c…/device_kernel.asc` 102 行）

- 定位：`grep -l 'void gdn_bwd_impl_kernel('` 命中（7263B，与判决同进程同 boot cache）——补 D3 §4.3"容器重编件"盲区
- R6 §3.1 逐条判据：
  - `db_acc` 清零：:51 每 token 一次、j 归约边界 0..8 ✓ 正确
  - `DBeta` store（:75）地址表达式与 `DV_`（:72）对称 ✓ 正确
  - **二次求值：:68 `uj = bt*(buf[j+104]-p)` 与 :73 `db_acc + du*(buf[j+104]-p)` 确为两次独立求值（未 CSE）**——R6 预测命中
  - 但 `p` 在两求值间**未被改写**（判据另一半未命中）；两次 load 间确有 `buf[j+72]/buf[j+80]` 写与 `DV_` GM store
- **判读（与 A 臂联立）**：源码层无可见错误；改写敏感（8.002e-01→5.997e-01）但消除重复求值不修好 ⇒ **候选 A 降级**：病灶对源码形态敏感、沉在 asc→机器码或执行层之下（候选 B/D 嫌疑上升）
- 附注（不强外推）：fork 侧 dbeta 静默错值与 TA 侧 507015 unaligned 都属 UB 访存/codegen 家族，症状不同；仅作 #1885 关联素材记录

## [7] B 线修复验证（诊断腿 2 `w11b`）：out_idx 修复成功，全链露出下一敌

- [D2-1] 卡上版本自证：修复版 sha `0a0ea3a2…`（out_idx=[1]/[5] 落卡）
- [D2-2] **solve_a 修复完全成功**：`nparams=8 result=[5] ws=[] agm=[6,7]`（原 result=[7]）；5 实参直调 `D2-A5-OK (1,1,64,64) torch.float16`——IndexError 病灶已除
- [D2-3] 全链小 case：**507015 芯片异常**（`VEC supports illegal configurations in commands`，subErrType 4，kernel `main_kernel` BlockDim [1]）——cumsum/solve_a 已过（同 cache 执行成功）⇒ 崩点在 **scan1/scan2a/scan2b 三者之一**（首次真机执行）
- 长腿（L0 全量/真形档）自动跳过；定位诊断腿 3 已发射（cache 复用）

## [8] 507015 定位与根因（诊断腿 3 + 生成码/TA 源码三级取证）

- [D3] 逐段外提：`D3-CUMSUM-OK / D3-SOLVEA-OK / **D3-SCAN1-FAIL 507015**`——**肇事 kernel = scan1**（首次真机执行）
- 键识别（w11b_cache 五键 kernel.cu 签名对照）：cumsum=0c7c31 / solve_a=8e5b01（含 2 agm ✓）/ **scan1=266a56** / scan2a=1ff2d4 / scan2b=10a5a9
- 生成码差异（scan1 266a56/kernel.cu）：**L268-269 `SetMaskCount(); SetVectorMask(0,4096)` 后 L286 `Sub(..., MASK_PLACEHOLDER, 64, ...)`——count 泄漏 17 行未复位**（64×4096 越界）；对照 solve_a（跑过）同形 Sub（L163-165）**有 SetMaskNorm 复位**
- 全链扫描（模式状态机）：**scan1 1 处、scan2b 21 处同类形态、cumsum/solve_a/scan2a 0 处**
- **根因链（TA 源码三级取证）**：
  1. `common.h:305-323`（本地 tilelang-ascend 仓，与 ta_pkg 同版）：`copy_ub_to_ub` 跨 dtype 走 `AscendC::Cast(dst, src, mode, calCount)`——**count 模式接口，执行后 mask 残留 count 态**
  2. `ascend_vector_mask_legalize.cc:199-200`：跨型 copy 声明为 `NormalFullPostState()`（:124-127，**错误地认为留下 norm/full**）→ `facts_` 被污染为 norm
  3. 后续需求 `kExact(norm)` 的 Sub 的 `Repair`（:570-578）误判已满足 → **不发射 SetMaskNorm** → Sub 继承 count=4096 以 repeat=64 执行 → 507015
  - 旁证：solve_a 有同 cast 对但中间有复位型指令（故它通过）；scan2b 21 处预示其真机也会炸
- **workaround（零 TA 重编）**：pass config `tl.ascend_vector_mask_reuse=False`（enum 见 `pass_config.py:55`）→ legalize :427-428/:437-438 **每条 terminal 前后 `facts_={}` 强制清空 → 所有 mask 设置强制发射**（保守化；多几条 SetMask 配置指令，无功能代价）
- 修复落盘：pass_configs 加第 6 key + H3 头注同步；**容器编译门 W11C-ALL-OK**（20/20 编译 + mask 扫描全 0，scan1 生成码 B 19843→22189 字节）
- [D4] 卡上验证腿发射中（逐段 → 全链 → L0 全量 → 真形档）

## [9] 阶段截断 bisect v1（`w11cut`，探针 `_w11_scan1_cut.py`，单进程 4 腿——仅 CUT1 为干净判决）

- **探针语义修正**：S2 段无 cut 门且实测未被 DCE（探针头注的前提落空）⇒ **CUT1 = S1+S2**（非"S1 only"）；CUT2 = CUT1 + DCE 锚、CUT3 = +S3、CUT4 = +S4
- **CUT4 键 == 生产 scan1 键**（`2482e77f…` 与 `w11d_cache` 全链编译键逐字相撞）⇒ cut=4 的 IR 与生产 scan1 完全一致
- 判决：4 腿单进程内全 FAIL 507015（首腿 aicore exception 后流污染，后 3 腿非独立；plog 佐证多次 launch failure）
- **对位（决定性）**：CUT1 生成码 136 行 **0 行独有**（逐字 ⊂ CUT4/生产 scan1 347 行）⇒ 截断无伪影，**炸点就在生产 scan1 的 S1+S2 指令序列内**
- 附带排除：CUT1 生成码中 S1 的所有形态（count=4096 fill / fp16↔fp32 cast / 同型 DataCopy）在 solve_a 各有 PASS 对照（L72-74 / L141 / L173）

## [10] 变体封杀序列（`w11cut2` 4 腿 + `w11cut3` 4 腿 + `w11pexp` 微探针；全部独立进程）

- **w11cut2（S1/S2 变体）**：V1(S1+S2 原样)**FAIL**、V4(S1 only)**PASS**、V2(窄拷→标量 SetValue)**FAIL**、V3(窄拷→对齐 8-lane)**FAIL**
  ⇒ **S1 全段无罪**；**窄拷形态换掉不足以救活**（窄拷非必要条件）
- **w11cut3（S2 微割）**：M1(E 链: fill/copy/Exp(e)/锚)**PASS**、M2(+glast 链)**FAIL**、M3/M4 级联 FAIL
  ⇒ **炸点在 glast 链**（窄拷 + `T.tile.exp(glast_ub)` + GM 锚）
- **w11pexp（微探针）**：P1(64-elem Exp repeat=1)**PASS**（数值 exp(0.5)=1.6487 ✓）；**P2(窄拷+8-lane mask=255 exp+落 GM) FAIL——9 行微内核复现 507015**
- 已锁定的两个候选形态（P2 内）：① 窄拷 `DataCopy(glast_ub[0], g_ub[63], 1)` @源 252B 非 32B 对齐；② `T.tile.exp(glast_ub, glast_ub)` 生成 `SetMaskNorm + SetVectorMask(0,255) + Exp(..., repeat=1)`（8-lane 位掩码 fp32）
- **待回判决（P3 窄拷纯 / P4 fill8+exp8 / P6 fill8 / P5 修复候选 glast 扩 [C]+64-lane exp）**——发射时 SSH 断连中断（`Connection closed ... port 31955`，ping 通），重连后补

## [11] plog 现场取证（`w11plog`，三变体进程 80672/83691/84783）

- 三进程 fault hash 各异（471740144446806136 / 12129929316329035612 / 9113336968250735719），fault kernel=main_kernel，task_id=6
- 错误现场（80672）：`errorStr: VEC supports illegal configurations in commands.`、`fixp_error0 info: 0x60000df, fixp_error1: 0xf5, subErrType:4`
- **横向对照**：窗九对齐家族 `fixp0=0x6000077`（errorStr "UB address ... not aligned"）、窗十 `0xd8/0xc`——**当前错误与窗九同 0x60000 前缀但不同细分**；错误域 = VEC（MTE errorCode=0 无 MTE 错）⇒ 属"配置面"拒绝（repeat/mask 参数），非窗九的地址对齐面

## [12] 炸点一修复与 scan1 真机 PASS（`w11pexp` 二轮 + `w11fix` + `w11fix2`）

- 微探针判决（独立进程）：P2(窄拷+exp8+GM落)FAIL、**P3(仅窄拷)FAIL**、P4(fill8+exp8)**PASS 1.6487**、P6(fill8)**PASS 0.5000**、P5(fill+窄拷+exp64)FAIL ⇒ **1-elem UB→UB DataCopy 为唯一触发形态**（4B<32B 粒度；exp8/fill8 无罪）
- 修复：scan1/scan2a/scan2b 六处 `T.copy(g_ub[..], glast_ub[0:1])` → 标量 `glast_ub[0] = g_ub[..]`（注释含证据边界："granularity hypothesis not directly proven"）；P7（标量+exp8+落GM）**PASS 1.6487**
- 容器编译门：`onelem_ub2ub=0` 全绿（6 配置含真形档 scan1=23759B；SetValue 出现）
- **真机逐段（`w11fix2`，独立进程）：cumsum PASS / solve_a PASS / scan1 PASS** —— scan1 首次真机通过
- 全链/全 L0 仍 507015（scan2a 起，见 [13]）

## [13] 炸点二：scan2a（UB 地址非对齐家族）——探针已备，实例停止中断

- `w11fix2` 逐段：**scan2a FAIL**（scan2b 级联非独立）
- plog 换型：`UB address accessed by the VEC instruction is not aligned`，**fixp0=0xd8/fixp1=0xc 与窗十逐字同值**（窗十该家族 workaround 证伪、真凶未定位）
- scan2a 生成码静态解剖（52009B）：alloc 34 项全 32B 对齐、常量索引 0 命中、表达式步长/stride 全过（两轮静态无实锤、效率低）⇒ 转段级实验
- 探针就绪（容器 parse 过）：`_w11_scan2a_cut.py`（机械生成器 `_w11_gen_scan2a_cut.py` 保证逐字保真；cut=2=A+B→14538B、cut=3=A+B+C+D+E→42602B，F 段剪除）、`_w11_red_probe.py`（RED8 编译 PASS；RED9 TA InternalError——生产同形态可编译，构型差异待查）
- 上卡车 `w11s2a_run.sh`（cut3→cut2→RED8，独立进程）**实例停止中断，恢复即发**
- 效率复盘：本炸点定位消耗 3 轮上卡往返（P2-P6→fix→逐段），静态扫描 2 轮低收益；恢复后一车打全（cut3/cut2/red8/red9 + scan2b 独立腿），不再单线往返

## [14] scan2a 炸点收敛与根因：spill readback copy stride=0（w11s2a / s2a2 / s2a3 / s2afix）

### 段级 bisect 判决链（独立进程，45s 轮询）
- cut2（A+B replay/Kt/beK/W）= **PASS** → A/B 洗清
- cut3（A+B+C+D，含 QK^T/P/accum init）= **FAIL**；cut4（+E per-BV 链）= **FAIL**
- RED8（64 列行和 reduce）= **PASS 32.0** → reduce 形态洗清
- **scan2b 独立腿（随机边界直接喂）= PASS outs=4** → scan2b 形态无罪；五 kernel 只剩 scan2a
- cut31 首跑：门控 `cut >= 31` 在 cut=31 时全 True（31≥3/31/4/5）⇒ **未剪任何段，跑的就是生产全链**；其 cache 键与 w11fix2 生产 scan2a 键**逐字相撞（a6fb3410，833 行）**——意外成为"生成器保真"的铁证

### 根因（生成码级）
- 833 行生成码中唯一零参数拷贝：`copy_gm_to_l1<half,64,64>(k_l1[0], workspace_1[0], 64, 0, 0, true)`（stride=0/rows=0，L160）
- `workspace_1` 是 CV pass 追加的第 9 个 workspace：C 段 liveness 下 `io_xk`（=io_cc 别名，[64,64] fp16 8KB）被 **spill 到 GM**，spill 回读 copy 参数生成退化
- 对照：其余拷贝均 `64, 64, 64`；K 的非 spill 路径（scan2b L1043 直拷）PASS
- 注：错误码家族仍为 unaligned（fixp 0xd8），与 copy 的畸形 stride 参数的因果链待上板复核

### 修复
- C 段 K 路径 `K→io_xk→xk_ub→io_xk→k_l1` 四跳 → **一行 `T.copy(K[...], k_l1)` 直拷**（scan2b 同形已验证）；消除 io_xk 在 C 段的 liveness，spill 与退化 copy 一并消失
- 生成器门控修正为单调档位（2<3<4<5<6），杜绝"全 True"陷阱；后续段级验证一律用新档位
- 上游素材 D5：spill readback 的 `copy_gm_to_l1` 生成 `(cols, 0, 0)` 非法参数，应修 codegen（写回 `(cols, rows, stride)` 或拒绝 spill 该 buffer）

## [15] scan2a 根因终定：2D Parallel 中 1D 行广播的 lowering 生成 4B 步进 VEC 地址

### 尾段判决
- cut3（QK only）编译器 TVMError（探针 DCE 边界问题，非生产代码）
- cut4（A+B+C+D）**FAIL 507015** ⇒ 炸点收敛到 C/D 段
- P10（exp 64×64，repeat=64）/ P12（gamma 三连+mul）/ P13（同型 4096 全拷）**全 PASS** ⇒ 孤立形态全无罪

### 根因（cut4 生成码 L273-277，327 行文件）
- `Sub<float, false>(aux_ub[64i], zero_ub[outer_broadcast_idx_2], aux_ub[64i], ..., BinaryRepeatParams())`——**src0 = `zero_ub + 4×i` 字节**，63/64 行为非 32B 对齐 VEC 地址 ⇒ fixp 0xd8 unaligned 家族
- 生产源码即 scan2a L648-649：`for (i,j) in T.Parallel(C,C): aux_ub[i,j] = zero_ub[i] - aux_ub[i,j]`（**1D 行广播进 2D Parallel**）
- **对照**：scan2b 同型（687 行 L406 逐字同形）PASS 因在 `T.serial` 标量循环（S7g "no broadcast temp" 已规避）；scan1 只有 1D 版（`Sub(th[0], zero[0], th[0], repeat=1)` 基址对齐安全）
- 机理闭环：同形态 + 不同 lowering 路径 = 不同命运

### 修复（5 处，一元负号，向量 Muls）
- scan2a C 段 L648-649 / F 段 L852-854：`zero_ub[i] - x[i,j]` → `-x[i,j]`（注释带机理）
- scan2b L1186/L1216/L1252（serial 内）：同上（防御，保持 serial 结构）
- 1D 版（L368/601/1047 `th_ub[i] = zero_ub[i] - th_ub[i]`）**保留**（scan1 真机 PASS 已证安全）
- 上游素材 D6：2D Parallel 中 1D 行广播的 lowering 应转 Broadcast 或报错，不得生成 4B 步进基址

## [16] 全链真机 PASS（w11fix3，历史性）

- **LEG-cumsum / solve_a / scan1 / scan2a / scan2b 全 PASS**（scan2a、scan2b 首次真机通过）
- 新生成码：scan2a 815 行（3e02b969）、scan2b 835 行（3602fa8a），**rowbroadcast_sub=0**（4B 步进 Sub 消失）
- 全链由三轮修复叠加达成：[12] glast 标量化 + [14] C 段 K 直拷（消 spill）+ [15] 5 处一元负号（消 2D 行广播）
- 后续：L0 六梯度精度 → 真形档 workspace → 窗口三收

## [17] L0 首跑：从"炸"到"数值差"的阶段跨越（w11l0）

- **7 个 L0 case 全部跑完无崩溃**（此前均 507015）——kernel 链路全通
- CPU 双门 PASS：adjoint 5.999e-10、chunk-math 全 1e-15 级（golden 可信）
- **ITEM 分解（l0_special fp32 h0=on）**：**dk/dbeta/dg/dh0 全 PASS（0.0）**（special mask 判据）；**dq=2.125 FAIL、dv 数值 0.0 但 mask FAIL**
  ⇒ **scan1（dh0）/scan2b（dk/dbeta/dg）数值路径正确，炸点收窄到 scan2a 的 dq/dv 输出**
- basic/typical/tail 多 case 输出 **nan**（fp16 basic：dv=4.85e-1、dh0=8.48e-1 非 nan 但错）
- 下一步：diag 逐 box（ws_g/ws_a/ws_h/scan2a 九个边界 box）定位 nan/inf 首现点

## [18] 数值诊断：NaN 首现于 scan1 输出，执行层脏读嫌疑（w11diag）

- **diag 逐 box（l0_basic fp32）**：ws_g（cumsum）=干净（absmax 26.0，mean -13.4）；ws_a =干净（absmax 1.0）；**ws_h（scan1）= 4096 个 nan**（全块）
- 下游全污染：s2a 全 box 带 nan（out5 degam 为 0）、s2b 全带 nan——**唯一根 = scan1**
- scan1 生成码（351 行）同步审计：AIC/AIV 的 CrossCore flag 7/8/9/5/4/6/3/1/2/0 **全配对**；各 workspace 写-读顺序正确；数据流静态推演全有限（E/theta ∈ [5e-12,1]，Vnew=U0-WH，H_out=Vnew^T Kt）
- 静态无 NaN 源 ⇒ **执行层脏读嫌疑**（未初始化/半写读；`Duplicate` count 模式 4096 的 repeat=1 均被随后 cast copy 覆盖，排出）
- **micro 车（S=0、K/V 常数、beta=1、h0=0，预期 H_out=(U0)^T K 全有限）已写，因实例重建（IP 变更）中断，恢复即发**
- 注：l0_special 的 dk/dbeta/dg/dh0 “PASS” 为 special mask 相等（双方 NaN）所致，非真通过

## [19] fp32 路径全炸、fp16 全干净；定位到 fp32 plain GEMM（w11mic8 + w11gp）

### micro 8 腿一车（scan1）
| mode | dtype | 输入 | H_out 判决 |
|---|---|---|---|
| 1 | fp16 | S=0 常数 | **PASS**（3.1e-2，量级正确） |
| 2 | fp32 | S=0 常数 | **inf** |
| 3/4 | fp32 | 随机 / S 随机 | nan 4096 |
| 5 | fp16 | 随机全 | **PASS**（absmax 5.7） |
| 6/7 | fp32 | h0≠0 / T=128 | nan 4096 |
| 8 | fp32 | 双跑确定性 | **nan mask 逐位相同、max_diff=0** ⇒ 非竞态 |

⇒ **fp16 数值路径全对；fp32 全炸（常数输入也 inf）**——不是溢出，是 fp32 特化路径的确定性错值。

### GEMM 微探针（64³，与 numpy fp64 golden 对比）
- **GP0 fp32 plain（A@B）**：nan=128 inf=139——**炸**
- **GP1 fp32 transpose_B**：**PASS max_abs_err=3.9e-06**——干净
- GP2 fp32 transpose_A / GP3 fp16 transpose_A：待回
- ⇒ **fp32 的 plain（非转置）L1→L0 加载路径是根因**（transpose_B 的 zT 布局正常）
- solve_a 的“干净”是**输入特殊（单位阵/三角初值）掩盖了 GEMM 缺陷**，不能作为 fp32 GEMM 正常之证
- 修复候选：① fp32 case 的 plain GEMM 改走 transpose_B 形态（UB 转置预存，保 fp32 全精度）；② 算子侧强 cast fp16（损精度）；③ 上游修 Catlass TileCopyTla 的 float zN 路径

### GP2/GP3 补全（一车四腿齐）
- **GP2 fp32 transpose_A**：nan=438 inf=898——**炸**
- **GP3 fp16 transpose_A**：nan=0 inf=0 但 **max_abs_err=4.35e+01（gold 2.9e+01）——数值全错**
- ⇒ **TA 的 GEMM 四形态全表（实测）：transpose_B 干净（fp32 3.9e-06）；plain fp32 炸；transpose_A fp32/fp16 均错**
- **推论**：scan1 fp16 的 H_out “无 nan” 不等于数值对（其 gemm4 走 transpose_A）——**micg 车（golden 对比）待回**
- 精度表：fp32 目标 rel ≤ 2e-4（TOL_F32），fp16 中间精度不够 ⇒ 绕行必须保 fp32（UB 转置 + transpose_B）
- 环境事实：本地 `3rdparty/catlass` 为**未初始化 submodule**（上游修复需拉源码或从 ta_pkg 取头文件）

### golden 对比车（w11micg，三腿）
- **mode1（fp16 常数）：vs fp64 golden max_abs_err=3.05e-05、rel=9.77e-04**（fp16 容差 2e-3）——**fp16 数学正确**
- **mode2（fp32 常数）：inf**（golden rel=inf）——fp32 炸
- mode5（fp16 随机）：micro 裸 randn 不符 test 的 `_np_sample` 分布，fp64 golden 自身爆 2.5e16——**该腿结论不可用**（不影响 mode1 定论）
- **定调：fp16 链路数学已通（数值+形态双绿）；fp32 阻在 GEMM plain/transA 路径**
- GP3（fp16 transA 探针错）不构成 TA 缺陷证据：探针 A 的 L1 布局与 transpose_A 期望的 zT 不一致；真实用法（scan1 mode1）已证数值对

### GP4/5/6（fp16 plain 对照 + fp32 绕行验证）
- **GP4 fp16 plain：PASS 2.28e-06**（fp16 全形态可用）
- **GP5 fp32 UB转置+tB（绕行）：PASS 4.19e-06**——**算子侧绕行技术路线验证成立**
- GP6 fp16 同路径：PASS 2.28e-06

### catlass float zZ 特化定位（本地拉 submodule 67fd2820）
- `copy_l1_to_l0b.hpp` 的 float 特化（`GemmType<float, layout::zZ, B1> -> <float, layout::nZ, B2>`）用 `LoadDataWithTranspose`，其 `repeatTimes`/循环界用 `C0_NUM_PER_FRCATLASSAL=16`（fp16 口径），**fp32 的 C0 应为 8**（`BYTE_PER_C0/sizeof(float)`）——半宽嫌疑
- 上游重编 libtvm 需完整 TVM C++ 源（ta_pkg 只有 python）——**不可行**；但 **catlass 头在 ta_pkg/3rdparty 且 kernel 编译时 include** ⇒ **头文件级 patch 可生效**（新 cache dir 避缓存）
- **patch 车已发射**：变体 A（repeatTimes 16→8）/ B（全换）/ C（repeatTimes+循环），各跑 GP0/1/2，结束自动恢复原头：`w11patch_fire.sh`

### 工程量盘点
- 全算子 `T.gemm_v0` 共 43 处：tB 11 / plain 20 / transA 12（若头文件路线不通，算侧绕行需改 32 处）

### patch 车判决（A/B/C 三变体均无效）
- **变体 A（repeatTimes 16→8）GP0 结果与原始逐位相同（nan=128 inf=139）**——该 zZ 特化根本未被使用
- 真 dispatch：`common.h` 的 `LayoutL1 = layout::zN`（非 zZ），plain B 走 `TileCopyTla` L433（zN→nZ, `LoadData ifTranspose=true`）；该特化已按 dtype 正确参数化（`ELE_NUM_PER_C0 = 32/sizeof(T)`）
- zZ 特化（L63，用 `C0_NUM_PER_FRCATLASSAL=16`）与本算子无关；但**上游确实存在两处 float 特化用 fp16 口径常量**（zZ→nZ、nN→zZ），属独立缺陷
- **结论：fp32 plain/tA 的炸点在 `LoadData ifTranspose` 的 fp32 指令级行为**（很可能是 16×16/fp16 语义限制），头文件参数微调无效；**上游修缮需 Ascend 指令级排查，本轮不做**
- **窗十一收口路线：fp16 全绿交付；fp32 标为 TA 缺陷（D7），算侧绕行方案已验证（GP5）但 32 点改造列入下一窗**

### L0 全量七 case 车（w11l0b，cache 复用，~8 分钟跑完）
- CPU 双门全绿（adjoint 6.0e-10 / chunk-math 2.8e-15）
- **fp16 两个 case 也 FAIL**（非预期）：`l0_basic fp16` dq/dk/dv/dbeta/dg = **nan**，dh0=8.48e-01；`l0_typical fp16` 同形（dh0=7.55e-01）
- l0_special fp32：dk/dbeta/dg/dh0 PASS 为 special mask 相等（非真值），dq=2.13 FAIL、dv max_abs=0 但 special mask 不符 FAIL
- fp32 五 case 全 nan/FAIL（D7 GEMM 上游为主因）
- **结论：scan1 数学已证对（mode1 golden 9.8e-4），nan 首现于 scan2a/scan2b（fp16 也炸）——新一层缺陷，待 fp16 逐 box 统计定位（w11diag16 车已发）**

### fp16 逐 box 判决 + Gamma 符号根因（D8）
- **逐 box**（w11diag16，fp16 basic）：ws_g/ws_a/**ws_h 全干净**（ws_h absmax 0.41）；nan 首现 **scan2a 的 DQ(out0,42)/WS_DW(out3,64)/WS_DKT(out4,42)/WS_DK(out8,40+2inf)**；typical(T=256) 更重（WS_DV 4090 nan+134 inf）
- **fp32 basic：ws_h 全 nan**（scan1 就炸，即 D7）——两镲层缺陷独立
- **根因**：`Gamma = exp(gam_r - gam_c)`（design L49、CPU golden test:237）——kernel 在 `g[i]-g[j]` 后又取负，变成 **exp(gam_c - gam_r)**：下三角差值最大 +32 → Γ ~1e13 → fp16 的 P 溢出 inf、mask 乘后 nan
- **5 处取负（均删除）**：scan2a P 组装、scan2a dA 重算、scan2b B11 dbeta、scan2b dX、scan2b B12 dgam；保留 3 处 1D theta 取负（L368/601/1047，语义正确）
- 取负来源：旧写法 `zero_ub[i] - aux_ub[i,j]`（当初为避开 2D 广播而写），修正编译错误时保留了取负语义 —— 双重负号
- **修复落盘**：gdn_bwd_v2.py sha bcb68a4b，py_compile OK，残留 0；**L0 全量重验车已发（w11l0c）**

### Gamma 修复后 L0 重验 + scan2a 逐 box 对比（w11l0c / w11s2ab）
- **L0（fp16）**：nan 全消失，dq/dk/dv/dbeta/dg/dh0 均为有限值但数值错（abs 0.48~4.1）——**Gamma 修复生效**；fp32 五 case 仍 nan（D7 未解）
- **scan2a 逐 box golden 对比（单 chunk，fp16）**：
  | box | rel | 判 |
  |---|---|---|
  | out6 DGAM2（dP⊙M⊙raw，**无 Gam**） | **6.7e-02** | 基本对 |
  | out0 DQ / out1 DH0 / out2 WS_DV / out8 WS_DK | ≈1.00 | 全错（特征等同输出≈0） |
  | out3/5/7（gold 恰为 0）、out4（gold 0） | — | 无信息 |
- **判读**：含 Gam 的 box 全错、不含 Gam 的 DGAM2 基本对 ⇒ 误差在 **Gam 参与的 P 组装及其下游**（非 GEMM 本体；scan1 ws_h 已证干净）
- **待查**：kernel 内 P/dqkt/dvnew 等中间量需内部 dump（外部 box 已无法再细分）；fp32 被 D7 遮挡
- 工具：`_w11_s2a_boxes.py`（fp64 cache 重放 B1-B13 对九个 box）

### 判别车 + 映射修正（w11s2ad / w11s2af）——**前一轮 box 映射是错的**
- **g0 车（Gam≡1）：DQ/DH0/WS_DV/DK 仍 rel≈1.0 ⇒ 误差不在 Gam**
- **发现映射错误**（靠读 scan2a 写点确证）：WS_DV 存的是 **dVnew(B1)** 不是 dv；WS_DGAM2 仅 B3；WS_DK 仅 B4'；WS_DE 仅 B5；WS_DGAM 仅 B9 scalar（单 chunk=0）
- **重写 golden 后（h0=zero / h0=on 两腿）**：
  | box | rel | 判 |
  |---|---|---|
  | WS_DE（h0=on） | **1.98e-04** | **对**（dO⊙QH rowsum） |
  | WS_DGAM2 | **4.0e-02** | 基本对（dP⊙M⊙raw） |
  | DQ / DH0 / WS_DV / WS_DW / WS_DK | ≈1.00 | 全错 |
- **判读**：dP、M、QH、dO、ones-rowsum 均对；错的是 **dQKt 经 `da_ub→io_cc→workspace_vcc→da_l1` 这条 UB→GM→L1 链后参与的 GEMM**（DQ=dqkt@K、DK=dqkt^T@Q），以及 P 同型链的 WS_DV=P^T dO——而 DE/DGAM2 不经该链∴对
- 下一步（纯本地）：在 kernel 内把 dqkt 的 **L1 内容** dump 到 GM 对比，区分“链传输错”与“GEMM 错”；改完才需一车验证

## [24] 本地改动（无需实例）——已完成两项

### L1 GEMM 探针 mode=7（dqkt/K 实测）
- `_w11_gemm_probe.py` 新增 mode=7：用 fp64 golden 算出的 **dqkt 与 K** 直接喂 64³ plain fp16 GEMM，与 `dqkt@K` 对比
- **本地预演（--golden-only，无卡）**：`dqkt_max=1.275e+01`、`dqkt_nonfinite=0`、**`hf16_overflow=0`**；`gemm_max=3.909`——**dqkt 在 fp16 可表示范围内，排除“dqkt 本身溢出”假设**
- **判读表（上卡一腿）**：mode=7 PASS ⇒ GEMM 本体对，误差在 `da_ub→io_cc→workspace_vcc→da_l1` 链；mode=7 FAIL ⇒ GEMM 对含 M-mask 的稀疏/零元素数据错

### L2 D7 绕行的 UB 容量方案（静态分析完）
- `_s7f_ub_budget.py ../gdn_bwd_v2.py` 实测：scan1 **169.5 KiB（fp32 上界 209.5，无复用静态值已超 192，实际靠 liveness 复用才编过）**；scan2a 158.8/166.8（余 25）；**scan2b 175.1/183.1（余 9，不足 16）**；solve_a 112.8/128.8
- **约束**：绕行**不得新增 alloc**，必须逐点复用当刻空闲的 UB（如 scan1 的 `hin_ub` 32KB 在 Kt/bek/bv 转置时刻空闲；需逐点确认 liveness）
- **开关**：绕行只在 `dtype == "float32"` 编译期分支启用，**fp16 零影响**（fp16 已证全对）
- 32 点（plain 20 + transA 12）的逐点复用表需逐点 liveness 分析，列入下一窗

### 下一车窗清单（待实例恢复，一车打全）
1. `_w11_gemm_probe.py 7`（判读见上）；若 PASS 则转向 dump `da_l1` 内容
2. `h0=on` 的 `out2/3/8` 再判（本轮已得 rel≈1.0，但需确认非 fp16 舍入）
3. fp16 全量 L0 复跑（确认本轮无回归）
4. （可选）D7 绕行 scan1 三点的试点上卡验证

## [25] 计划修订（2026-10-18，实例已停）：本地优先分组

### 局势重述
- 已知：P 与 dqkt 同走 `da_ub→io_cc→workspace_vcc→L1` 链；两者下游（dvnew=P^T dO、DQ=dqkt@K、DK=dqkt^T@Q）全错，而链前的 `da_ub`（DGAM2）与不经链的 DE 均对
- 已知：scan1 走同型链（xk1→io_xk1→GM→bek_l1/w_l1）且数值对（golden 9.8e-4）
- **⇒ 最高优先级假设：scan2a 的链写法/复用与 scan1 有实质差异**——纯本地可逐字对照

### A 组：本地静态取证（本轮推进，零实例）
- **A1** 链写法逐字对照：scan1 的 xk1/bek 链 vs scan2a 的 P/dqkt 链（copy 参数、中转 box、是否同 buffer 复用）
- **A2** `workspace_vcc` / `io_cc` 的读写点与生命周期审计（跨 bv 迭代覆盖、多消费者共用）
- **A3** 疑点清单 + 修复候选（仅在 fp32/fp16 分支内改，fp16 已验证项不动一旦改则 L0 必须全量回归）
- 产出判据：A1/A2 每条疑点必须带**行号 + 为何 scan1 不中招**两要素，否则不入清单

### B 组：等实例（一车打全，判读表已定死）
- **B1** `_w11_gemm_probe.py 7`：PASS ⇒ 链错；FAIL ⇒ GEMM 对 mask 后稀疏数据错
- **B2** `h0=on` 的 out2/3/8 复判（排除 fp16 舍入）
- **B3** fp16 全量 L0 回归
- **B4** （可选）D7 绕行 scan1 三点试点：UB 不得新增 alloc（scan1 fp32 上界 209.5 已超）

### 延期
- 新建 openspec 子变更（把 gdn_bwd_v2 纳入多级计划）：待 A 组结论后再定，避免在根因未明时固化任务面（R29：验收面先于实现——任务面应建立在已定位的根因上）

## [26] A 组本地取证结果（A1/A2/A3 完成，零实例）

### A1 链写法逐字对照（scan1 ✓ vs scan2a ✗）
| | scan1（对） | scan2a（错） |
|---|---|---|
| beK 中转 | `workspace_vdk` | `workspace_vcb` |
| bv 中转 | `workspace_vcb` | `workspace_vcb`（**与 beK 共用**） |
| 链形态 | UB→UB→GM→L1 | 同 |
| GM→L1 直拷 | 无（全走链） | hin/q/k 直拷 |
- 写法无实质差异；共用 vcb 但 beK 的读（L624）在 bv 写（L694）之前，**顺序正确**

### A2 workspace 复用审计
- `workspace_vcb` 在 scan2a 复用 7 次、`workspace_lcb` 5 次、`workspace_ldk` 5 次——**全部严格 1W1R 紧邻成对**，无跳代覆盖
- **DE（经链 + ones-GEMM）rel=2e-4 正确** ⇒ **链与 GEMM 形态均非病灶**
- **⇒ 排除：链传输错、workspace 覆盖、GEMM 形态错**

### A3 fp16 精度上界（新工具 `_w11_fp16_emul.py`，纯 CPU）
理想 fp16 pipeline（每张量按 kernel 存储 dtype 舍入、GEMM 精确）：
| box | emul_rel（正确 kernel 的上界） | 卡上实测 | 倍数 |
|---|---|---|---|
| DQ | 3.9e-04 | 1.0 | ~2500× |
| DH0 | 1.2e-04 | 1.0 | ~8000× |
| WS_DV | 2.9e-04 | 1.0 | ~3400× |
| WS_DK | 3.6e-04 | 1.0 | ~2700× |
| WS_DGAM2 | 1.7e-04 | 4.0e-02 | ~240× |
- **⇒ 确凿是 bug，不是 fp16 精度**；同时排除了“需改 fp32 accum / bf16 混合”的整条设计分支

### 已排除清单（下一窗不必再试）
1. Gam 符号/计算（g0 车：Gam≡1 仍全错）
2. dqkt 溢出 fp16（max 12.75，hf16_overflow=0）
3. 链写法差异、workspace 覆盖、GEMM 形态
4. fp16 存储精度（上界 1e-4）

### 剩余唯一路径（需实例）
- **mode=7**（GEMM 本体 vs 链）——但 A2/A3 已使“链错”假设降级，mode=7 若 PASS 则需重新设计假设
- **kernel 内 dump**：dqkt（da_ub）与 dqkt 经链后（da_l1）的内容对比——**唯一未直接观测的环节**
- 提示：A2 已证“链 + GEMM 各自都对”（DE），故 dump 时要同时 dump **GEMM 的两个输入**（da_l1 与 k_l1）——单 dump 一个无法区分“输入错”与“GEMM 读错布局”

## [27] 上机就绪（2026-10-18 收口，三件齐备）

### 1. kernel 开关（`gdn_bwd_v2.py`）
- 新增 `_S2A_DBG = os.environ.get("GDN_S2A_DBG", "0") == "1"`；开启时 scan2a 把 **dqkt（da_ub）** 写入 `WS_DKT` 槽位（单 chunk 下该槽生产值为 0，不污染结果）；**默认关，生产路径走 else 分支逐字节不变**
- 本地：`py_compile` OK；`grep -c _S2A_DBG` = 4

### 2. driver（`_process/_w11_s2a_dbg.py`）
- 在 import gdn_bwd_v2 前设 env；调用 cumsum→solve_a→scan1→scan2a；读 out4（=dqkt）与 out0（DQ）对 golden
- **本地 golden-only**：`dqkt max=1.275e+01`、`dq max=3.909e+00`（与 L0 basic fp16 一致）

### 3. fire（`.tmp_a1/w11dday_fire.sh`）——实例恢复即发
- TA 重铺：`refs/ta_pkg/ta_pkg.tar.gz`（36,478,146 B，已验证存在；实例重启后 /tmp 清空自动重建）
- 三腿：**LEG1** s2a-dbg / **LEG2** gemm mode7 / **LEG3** h0=on boxes；**每腿独立 TILELANG_CACHE_DIR**（`GDN_S2A_DBG` 不在 TVM 缓存键内，旧 cache 会静默复用非 dbg 产物）
- **LEG4**（同日补）：`_w11_num_diag.py` fp16 basic/typical + fp32 对照——**nan 回归门**（Gamma 修复后 fp16 应 nan-free），与定位腿同车兼得
- `bash -n` 语法门通过

### 判读表（已定死）
| LEG | 结果 | 结论 |
|---|---|---|
| LEG1 | dqkt rel < 5e-3 且 DQ rel ≈ 1.0 | **GEMM/L1 读路径错**（dqkt 对、GEMM 错） |
| LEG1 | dqkt rel 大（≥ 4e-2 量级） | **dqkt 构造错**（dP/Gam/M 上游） |
| LEG2 | PASS | 交叉验证 dqkt@K 的 GEMM 本体在该数据上无问题 |
| LEG2 | FAIL | GEMM 对 mask 后稀疏/零数据有特异性缺陷 |
| LEG3 | out2/3/8 rel | 确认非 fp16 舍入（与 emul 上界 ~3e-4 对比） |

### 风险声明
- `_S2A_DBG` 默认关时 Python 走 else 分支，**预期生成码与改动前逐字节相同——但未上卡验证**（本地无法生成 ascend 码）。若下窗发现 fp16 回归，**第一嫌疑即这 3 行**，`git revert` 可立即隔离
- `WS_DKT` 槽位借用的前提是单 chunk（`dkt = vnew @ dp_rel`，dp_rel=0 ⇒ 0）；多 chunk 不可开该开关

## [28] ★ D11 根因锁定：L0C 超配越界（scan2a 196KB / scan2b 156KB vs 910B 128KB）

### 证据链
| 证据 | 数据 |
|---|---|
| 910B（A2）L0C 容量 | `codegen_ascend.cc:39` `ASCEND_A2A3_L0C_SIZE = 131072`（128KB） |
| codegen 行为 | 只 `pipe.InitBuffer(ascend_l0c, 131072)`，**不按 kernel 实际 alloc 总和检查**（L895-911） |
| **scan1（数值对）** | L0C = **98304B（96KB）< 128KB** ✓ |
| **scan2a（数值错）** | L0C = **200704B（196KB）= 1.53× 超限** ✗ |
| **scan2b（数值错）** | L0C = **159744B（156KB）= 1.22× 超限** ✗ |
| 独立 GEMM 探针（GP0-PASS / mode=7 / D10 累积探针） | 均只 1 个 L0C（16KB），**全部正确**（1e-7 级） |
| 输出残缺模式 | DQ 仅 1/64 列有值、dqkt 仅前 16 列、行 20+ 下溢为 0——**越界后 L0C 互相覆盖的典型形态** |

**⇒ 三个 kernel 的正确/错误与 L0C 是否超限完全对应；所有独立探针（单 L0C）均正确。**

### 修复方向
1. **算侧**：将 scan2a（13 个 L0C）与 scan2b（12 个）降到 ≤ 8 个（复用 dq/dk/dw/dkt 的半盒、dp/u0/dvnew 等），使总量 ≤ 128KB——**需逐点 liveness 设计，工作量中等**
2. **上游**：codegen 应对 L0C/UB/L1 的静态超配**报错而非静默越界**（与 D7 同属“静默错值”族）
3. **验证一车**：修后跑 LEG1 + L0 全量（fp16 两 case 应转绿）

### 本轮已排除清单（累加，下窗不重试）
Gam 符号（g0 车）· dqkt 溢出 · 链写法 · workspace 覆盖 · GEMM 形态 · fp16 精度（上界 1e-4）· L0C 累积（D10 正反两档均 1e-7）· ub→GM→L1 链（A2 审计）

## [29] D11 否决、**D12 锁定**：累积 GEMM 目标的 L0C→GM copy 退化为 1 列

### D11（L0C 超配）否决——生成码实测
`_w11_l0c_src.py` 打印 scan2a 生成码：`InitBuffer(ascend_l0c, 131072)`，实际 offset：dw0=0、dp=16384、u0=32768、dvnew=49152、qh=65536、rs=81920、dq0=86016、dh0=102400、**dkt0=0（与 dw0 复用）** ⇒ **实际峰值 118784B = 116KB < 128KB**，未超配。13 个同时活的 L0C 探针 SIGSEGV 是人为构造，不代表生产。

### D12（新，锁定）：`copy_l0c_to_gm` 倒数第二参数 1 vs 64
| L0C | copy 参数 | 对应输出 | 实测 |
|---|---|---|---|
| dp_l0 | **64** | DGAM2 | rel 4e-02 ✓ |
| qh_l0 | 64 | DE | 2e-04 ✓ |
| u0/dvnew | 64 | WS_DV | 链正确但上游 P 错 |
| **dq0_l0** | **1** | **DQ** | rel 1.0，**仅 1/64 列非零** |
| **dh0_l0** | **1** | **DH0** | rel 1.0 |
| **dw0_l0** | **1** | **WS_DW** | rel 1.0 |

**共同点：三个参数=1 的 L0C 恰好都是 `init=False` 累积 GEMM 的目标**（B4 累到 dq0、B6b 累到 dh0、B6 累到 dw0）；参数=64 的都是单次 init=True。
**⇒ D12：累积 (init=False) GEMM 目标的 L0C，其 copy_l0c_to_gm 的 burst/行参数被推成 1，只搬了 1 列——与“DQ 仅 1 列有值”逐字吻合。**
（D10 探针未复现：单 L0C、单输出的极简 kernel 上 layout 推断正常，需多 L0C/条件分支上下文才触发——待下一步 probe 确认。）

### 修复候选
1. **算侧绕行**：DQ/DH0/WS_DW 三处拆为“两个独立 init=True 的 L0C + UB 相加”，彻底避开累积 L0C
2. **上游**：修 codegen 对累积 GEMM 目标的 copy_l0c_to_gm 参数推断（与 D7/D11 同属“静默错值”族）
3. 验证：修后跑 L0 fp16 两 case（应转绿）+ LEG1

## [30] D12 修复尝试（DQ 一处）及其未果——实例停止收口

### 已落地的改动（本地，py_compile OK）
- B5b（hh==0）：`init=(bv==0)` → **`init=True`**（常量，避开累积）
- B4（hh==0）：`dq0_l0, init=False` → **`dp_l0, init=True`**（dp_l0 在 :850 后空闲）
- DQ 写回：改为 **两段 1W1R（dp_l0 / dq0_l0）+ UB 相加**（保持 workspace_ldk 严格 1W1R）
- `t1_ub` 复用已核 liveness（:786 后、:928 前无使用）

### 车结果（w11l0d，L0 全量，10s 轮询）
- **DQ 仍 FAIL rel=3.909**（与修复前相同），但 **ratio 0.0005 → 0.0002** ⇒ 生成的 kernel 确实变了，**但未修好**
- fp16 其余项未变；fp32 仍 D7
- **新生成码的 copy 参数未获取**（w11l0cs 检查车在实例停止前只到 LAUNCHED；artifact 仍是旧版 803 行、`dq0_l0` 参数=1）

### 结论与下一步（本地可做）
1. **D12 降级**：参数 1 vs 64 与正确性的对应性强，但“避开累积目标”并未修好 DQ ⇒ **另有原因或修复不完整**（可能是 dp_l0 的跨大段复用、或 DQ 的 0 型错误另有来源）
2. **下窗第一件事（一车）**：用 `_w11_l0c_src.py` 打**修复后**生成码的 copy 参数——若 `dp_l0` 的 copy 仍是 1 或 DQ 写回段引入了新拷贝，则 D12 仍成立但修复手法错；若已是 64 而 DQ 仍错，则 D12 仅是伴随现象，需转 dump dq0_l0/dp_l0 的 GM 内容
3. 本地先行：读生成码的 copy 语义（`copy_l0c_to_gm` 在 common.h 的最后两参），确认“1”到底是 burst 数还是 n 维——不靠猜
4. DH0 / WS_DW 两处**不动**（等 DQ 先定案，避免无验证的堆叠修改）

## [31] ★ D13 机制确认：realTailN 随循环变量退化为 1（已落修复，待上卡）

### 语义实证（读 common.h:188-214，不靠猜）
```
copy_l0c_to_gm(dst, src, realDstN=1, realTailM=0, realTailN=0, unitFlag=0)
  tailN = realTailN == 0 ? srcN : realTailN;   // realTailN=1 => 只搬 1 列
```
生成码里 `..., 64, 64, 1, 0)` 的倒数第二参就是 **realTailN=1**；`..., 64, 64, 64, 0)` 是 64。

### 规律（与正确/错误完全对应）
- **GM 切片含循环变量**（`hh * BK`，在 `T.serial(bk_num)` 内）⇒ realTailN 推为 **1** ⇒ 只搬 1 列：**dq0（DQ）、dh0（DH0）、dw0（WS_DW）— 全错**
- **切片无循环变量**（`:850` 的 `workspace_lcc[cid,:,:]` 等）⇒ realTailN = **64** ⇒ dp/u0/dvnew/qh — 全对/链正确
- 这也解释了为什么上一轮“把累积换成单次 init=True”无效：**DQ 写回仍在 T.serial 循环内**，realTailN 依旧 1

### 已落修复（本地，py_compile OK，待上卡）
三处输出写回循环 `for hh in T.serial(bk_num)` → **`for hh in range(bk_num)`**（Python 编译期展开，bk_num=1 时切片变常量）：
1. B15 的 DH0/WS_D 写回（含 dh0_l0）
2. WS_DW 写回（dw0_l0）
3. DQ 写回（dp_l0 + dq0_l0 的 UB 相加）

### 上卡验证（一车即可）
1. `_w11_l0c_src.py`：确认三处 copy 的 realTailN 变为 **64**（静态可判）
2. L0 全量：fp16 两 case 的 dq/dh0 应从 rel≈3.9/0.85 转为 PASS 量级
3. 若 dq 转绿而 dk/dv/dbeta/dg 仍错 ⇒ 继续按同法查 scan2b 的同类循环（B11/B12 的 dk/dgam 写回）

### 本地可做（无卡）
- 按同一规律静态排查 scan2b 的 L0C→GM 写回是否也在 `T.serial` 内（预估至少 2-3 处：dk0/dk1、dgam2、dA）

## [32] ★★ D15 实锤：2D Parallel 内 `g_ub[j]` 的 j 索引恒为 0（本轮真主因）

### 一车三腿（w11one）+ g_ub 车（两轮自证）判决
| 观察 | 数据 | 结论 |
|---|---|---|
| L0 fp16 | dq 3.909 / dh0 0.848（与修前逐位相同） | range 展开无效 |
| GEN 生成码 | 4 处 copy 的 realTailN **仍为 1** | D13 修法失败；规律改为“目标是 workspace_ldk ⇒ 1、workspace_lcc ⇒ 64” |
| DBG dqkt | **nz=64/64** | **D9 被否**（dqkt 不缺列） |
| DBG g_ub | card[:6] 与 golden **逐位相同**、absmax 2.601e+01 相同 | **D14 被否** |
| **DBG Gam** | **row10 card=[0.0092 ×6] vs gold=[0.0092, 0.0143, 0.0186, 0.0237, 0.0355, 0.0619]**；**diag card=[1.0, 0.6452, 0.4969, 0.3901]（应全 1）** | **D15 实锤** |

**机理**：`Gam[i,j]` 在 kernel 里算成 `exp(g_i - g_0)`——第 10 行每列相同、对角 0.6452 恰为 exp(g₁₋g₀)。即 **2D `T.Parallel(C,C)` 内读取 1D 数组 `g_ub[j]` 时第二个索引退化为 0**（与 D9 同族的 2D lowering 缺陷，但表现不同）。

### 为何之前没暴露 / 为何 solve_a “对”
- solve_a 的 `gm_ub[i,j] = g_ub[i] - g_ub[j]`（L236）同构；mode1（g≡0）下任何索引都等于 0 ⇒ **恰好对**；g≠0 的 case 同样错但未被单独验证
- scan1 的 H_out 在 mode1 下对（不依赖 Gam 的列索引）

### 修复候选
1. **外积形式（推荐）**：`Gam = exp(g) ⊗ exp(-g)`，用 ones-GEMM/外积把两个 **1D** 向量相乘，彻底避开“1D 数组的第二维索引”
2. **算侧“先物化 2D”**：用逐行 `T.serial` + 1D `T.Parallel` 把 `g_ub[j]` 写成 `grow_ub[j]`（每行复制一份）再参与 2D 运算——多 64 份拷贝但语义安全
3. **上游**：修 `T.Parallel`/代码生成对 2D 循环内 1D 缓冲索引的 lowering（与 D7/D11/D12 同属“静默错值”族）

### 影响面（一次评估）
- scan2a：P 组装、dA、dqkt 三处 Gam（均错）→ DQ/DH0/WS_DV/DW 全错
- scan2b：B11/dX/B12 三处 Gam（同样错）→ DK/DGAM/dg 错
- solve_a：gm_ub（同缺陷，g≠0 时 A 也错）→ 但 scan1 的 H 链在 mode1 下已验，其余档未验

### 下窗第一车（优先级）
1. 用“先物化 2D（方案 2，最小改动）”修 scan2a 的三处 Gam → L0 fp16 看 dq/dh0
2. 同步修 solve_a 与 scan2b 的三处（同手法）
3. 若修后 dq 转绿 ⇒ D15 定案；再回头看 D13（realTailN，作为独立第二层）

## [33] D15 修复生效（Gam 2.5e-08）+ D13 部分生效（dqkt 10%）+ DQ 新残差

### 已落地
1. **D15 修法（scan2a dqkt 段）**：减法改为 **双 `T.serial`**（scan2b S7g idiom）——`g_ub[i]`/`g_ub[j]` 均为标量读
   - 试错记录：serial 外层 + Parallel 内层 → **507015**（不可用）
2. **D13 修法（DQ 写回）**：`bk_num == 1` 时改用**全切片** `workspace_ldk[cid, :, :]`（显式 Python 分支），使 realTailN 保持 64

### 上卡判决（同一 DBG 车）
| 观察 | 修前 | 修后 |
|---|---|---|
| Gam row10 | card 全 0.0092（j 恒 0） | **与 golden 逐位相同** |
| Gam diag | [1.0, 0.6452, 0.4969…] | **[1.0, 1.0, 1.0, 1.0]** |
| Gam rel | 1.0 | **2.543e-08** |
| dqkt rel | 1.0（仅前 16 列有值） | **1.024e-01（全 64 列，值合理）** |
| DQ rel | 1.0（仅 1 列） | **1.0（但 nz=64/64，列已全）** |

**⇒ D15 定案（真主因）；D13 列数已修好；剩余两个残差：**
1. **dqkt 的 10%**——上游 dP（B2，4%）与 Gam 修复后的新算式误差，待分段
2. **DQ 的 1.0**——DQ 应 = dqkt@K（10% 误差）但实测差一个 golden 量级 ⇒ **DQ 写回/UB 相加另有错**（不是 dqkt 的 10% 能解释的）

### 下窗（一车）
1. **DQ 残差**：dump `io_xk` 出口（保真到 DQ 的最后一个 UB）与 `dqq@K` 的 golden 对比；（或先把 `(E⊙dO)@H_in` 项置 0 验证相加顺序）
2. **dqkt 10%**：拆 B2 的 dP 与 Gam 乘法顺序
3. **扫描全部 7 处 Gam**（scan2a 3 + solve_a 1 + scan2b 3）改用双 serial——**solve_a 的 g≠0 case 也会错**
4. 再跑 L0 fp16（预期 dq/dh0 大幅改善）

## [34] 本地收口（实例停）：D15 推广 4 处 + DQ 改直拷，可上机

### 改动
1. **D15 推广（全部 4 处需修点）**：
   - scan2a P 段（:665）、scan2a dA 段（:880）、scan2a dqkt 段 → **双 `T.serial`**
   - **solve_a（:237）** → 双 `T.serial`（此前只在 g≡0 下“碰对”，g≠0 时 A 也是错的）
   - scan2b 三处（:1252/:1279/:1314）**本就是双 serial** ✓ 无需改
2. **DQ 全 0 的修法**：B4 改用独立单发 L0C **`dq1_l0`**（L548 已 alloc），其 GEMM 后**立即 `T.copy(dq1_l0, t1_ub)`（L0C→UB 直拷）**；DQ 写回只做 `dq0_l0 → xk_ub` + UB 相加 + 写出
   - **同时绕开三件事**：GM 中转、realTailN=1、liveness 复用窗口
3. 试错记录：serial 外 + Parallel 内 → **507015**（不可用）；`slice(None)` → 未采用

### 上机一车（即发）
`bash .tmp_a1/w11one_fire.sh`（GEN/DBG/L0 三腿，每腿独立 cache，10s 轮询）
| 腿 | 判读 |
|---|---|
| L0 | **fp16 dq/dh0/dk 是否从 3.9/0.85 降到 PASS 量级**（主判据） |
| GEN | DQ 写回不再走 L0C→GM，重点看 `copy_l0c_to_ub`（dq1_l0→t1_ub）与 dkt 的 realTailN |
| DBG | dqkt/Gam 保持正确（回归）；DQ absmax 应从 0 变为非 0 |

### 若 L0 仍红（预判）
- dqkt 的 10% 残差（dP 的 B2 段）会直接体现在 dq 上（预期 rel ~0.1，而非 1.0）
- 若 dq 仍 ~1.0 ⇒ DQ 通道仍有断点，优先 dump `dq1_l0` 与 `t1_ub`（L0C→UB 后的第一个接点）

## [36] 本轮上卡总结：S1G 突破、D15 算侧无解（四形态均撞后端）

### ✅ 历史性正面结果（w11prod 车，双 serial 版）
| 量 | 结果 | 意义 |
|---|---|---|
| **solve_a 的 A** | **rel=5.266e-05**，rowsum 逐位一致 | **g≠0 下 A 首次正确**（D15 在 solve_a 修好） |
| **scan1 的 H 链** | **rel=5.095e-04 < 5e-3 ⇒ PASS** | **前向链路端到端首次通过**（此前只在 g≡0 验过） |

**⇒ 这说明“D15 修好 ⇒ 前向正确”的因果链成立；问题只剩“怎么在不超时的前提下算对 Gamma”。**

### ❌ D15 的算侧四种形态全部失败
| # | 形态 | 结果 |
|---|---|---|
| 1 | 完整双 `T.serial`（4096 次/Gamma） | **能跑但 507014 超时**（L0 全 7 case） |
| 2 | `T.copy(g_ub, grow_ub[0,:])` + `T.tile.broadcast` | **507015** |
| 3 | `grow_ub[0, j]`（2D 源列索引） | **507015** |
| 4 | 双 serial + `if j <= i`（只算下三角） | **507015** |

**⇒ 当前代码回退到“Plain 2D”（能跑但 Gamma 仍退化），并在两处加 `D15 KNOWN-BUG` 注释指向上游 issue 草稿。**

### 下一步（唯一可行方向）
1. **方案 6（下窗首试）**：scan2a 只算 **1 次** Gamma（双 serial）存到独立 `gam_ub`（+16KB，175/191.75 KiB），dA/dqkt 段**复用**它；solve_a 1 处双 serial——总量从 12288 降到 8192 次，看是否逃出超时
2. **若仍超时**：上游修复（issue 草稿已备）；或尝试 `T.tile.gather`/`T.tile.duplicate` 等替代列物化 API
3. **独立于 D15 的进展**：DQ 的 copy 参数已验证为 `(64, 0, 0)`（全搬），带 dq1_l0 直拷的版本在能跑后会直接给出 DQ 的真值

## [37] 方案 6 已落地，但编译疑似卡死（本轮未能验证）

### 代码状态（已提交，py_compile OK）
- **scan2a**：Gamma 只在 P 段算一次（双 serial + exp）→ `gam_ub`；dA/dqkt 段直接复用 `gam_ub`（不再第二遍）
- **solve_a**：Gamma 用双 serial（S1G 已证该单点可接受）
- budget：scan2a **174.8/191.75 KiB（91%）**、solve_a 129 KiB——均未超

### 车结果（w11l0b，41 分钟无 ITEM）
- `/tmp/w11l0b_l0.log` 仅 1212B（import 阶段的 warning），**mtime 停在 18:36，9 分钟未更新**
- 实例侧进程状态未能确认（ps 命令被 talu 截断）
- **⇒ 既不是 507014 也不是 507015，而是“无输出”**：疑点指向**编译阶段**（gam_ub 的 [C,C] + 双 serial 可能让某 pass 变慢/展开膨胀）

### 下窗第一件事
1. **单独编译计时**：只 `get_kernel_source()` scan2a（不跑），量出编译耗时；若 >5 分钟即定位到编译问题
2. **降规模试**：把 `gam_ub` 换回 `aux_ub`（P 段算完就存 aux，dA 段不再需要 aux 的 raw 用途时优先保 Gam）——省 16KB 且少一个 buffer
3. **若编译也因双 serial 膨胀**：回到 plain 2D（已知可跑）+ 接受 Gamma 错，等上游修

### 本轮交付判据
- **可宣称的**：S1G 的 A（5.3e-5）与 H 链（5.1e-4）PASS；issue 草稿就绪；四种算侧形态的失败已固化
- **不能宣称的**：方案 6 的数值结果（未跑到）

## [38] 方案 6/7 实测：一个超时、一个编译卡死——已回退（六形态穷尽）

### 实测
| 方案 | 做法 | 结果 |
|---|---|---|
| **6** | 独立 `gam_ub`（[C,C] fp32，Gamma 算一次复用），UB 91% | **41 分钟只产出 507014**（能跑、超时） |
| **7** | `aux` 携带 Gamma、`dhp_ub` 携带 M（不新增 buffer），UB 83% | **40 分钟零输出**（`grep ITEM\|507014\|507015` = 0）——**连 kernel 都没进，卡在编译** |

**⇒ 方案 7 比 6 更差：去掉 16KB 反而从“运行超时”退化为“编译卡死”**——说明触发点不是 UB 占用量，而是**某个 pass 在 `aux` 跨段复用上的行为**（`aux` 同时承担 Gamma 与 dA 的乘数）。

### 六形态全表（D15 算侧已穷尽）
| # | 形态 | 结果 |
|---|---|---|
| 1 | 完整双 serial | 507014 超时 |
| 2 | `[1,C]` + `T.tile.broadcast` | 507015 |
| 3 | `grow_ub[0,j]`（2D 源列索引） | 507015 |
| 4 | 双 serial + `if j<=i` | 507015 |
| 6 | 独立 gam_ub + 一次 Gamma | 507014 超时 |
| 7 | aux 携带 Gamma + dhp 携带 M | 编译卡死 |

**⇒ 回退到 `4454abc7`（plain-2D Gamma + DQ 经 dq1_l0 直拷），该版本能端到端跑通。**

### 第七种形态（下窗首试，尚未验证）
**用 GEMM 做外积**：`Γ = exp(g) ⊗ exp(−g)`
- `e_l1[C,16]`：第 0 列 = exp(g_i)，其余 0；`einv_l1[16,C]`：第 0 行 = exp(−g_j)，其余 0
- `gemm_v0(e_l1, einv_l1, gam_l0)` → `[C,C] = exp(g_i − g_j)`（K=16 满足静态对齐要求）
- 再 `L0C→UB` 直拷后参与乘。**代价**：一个额外 [C,C] L0C（16KB），需确认 liveness 能把它与早期已死的 L0C 复用（当前峰值 116/128 KiB）

## [39] 方案 8（退化+转置+减）也 507014 —— 七形态穷尽，D15 只剩上游

### 方案 8（纯向量化，零标量循环）
利用 D15 的退化本身：`aux = g_i - g_0`（卡上实际值）→ `transpose` 得 `g_j - g_0` → 相减得 `g_i - g_j`。
- 代码：2 处 `T.tile.transpose(dhp_ub, aux_ub)`、0 处 serial Gamma、budget 159.1/191.75（83%，无新 buffer）
- **结果：L0 七 case 全部 507014**（逐位同方案 1/6 的失败形态）

**⇒ 一个重要推论：纯向量化的形态也超时，说明 507014 的来源不是“标量循环太多”，而是某个被引用的 2D 逐元素/transpose 段本身在 910B 上超出了 aicore 的单次执行预算。这与 UB 压力（方案 7 在 83% 下编译卡死）一起，把“算侧通过减负/换写法绕过”的空间彻底封死。**

### 七形态全表（最终）
| # | 形态 | 结果 |
|---|---|---|
| 1 | 完整双 serial | 507014 |
| 2 | [1,C] + T.tile.broadcast | 507015 |
| 3 | grow_ub[0,j]（2D 源列索引） | 507015 |
| 4 | 双 serial + if j<=i | 507015 |
| 6 | 独立 gam_ub（91%） | 507014 |
| 7 | aux 携带 Gamma + dhp 携带 M（83%） | 编译卡死 |
| 8 | 退化 + transpose + 相减（纯向量化） | **507014** |

**当前已回退到 `4454abc7`（plain-2D + DQ 直拷），该版本 L0 能跑完（数值因 Gamma 退化而错）。**

### 关于“NPU 使用率满”
实测：每轮超时测试会让 AICore 在超时窗口内保持高位（507014 后的等待/重试）。已在 fire 里把 timeout 从 2400s 收到 1500s；下窗应再加“10s 轮询发现 log 超 5 分钟无新行则主动 pkill”。

## [35] 一次收全（5 腿）+ 预案树

### 一车（`w11final_fire.sh`，每腿独立 cache，10s 轮询）
| 腿 | 新信息 |
|---|---|
| GEN | 所有 `copy_l0c_to_gm/ub` 的实参（D13 是否在各处生效） |
| DBG | dqkt/Gam/g_ub 的跨度 + **DQ absmax（直拷是否非 0）** |
| **S1G（新）** | **solve_a 的 A + scan1 H 链 vs fp64，在真实 g≠0 上**（此前只在 g≡0 验过） |
| S2AB | 9 个 box 的 rel（DGAM2 的 dP 残差、DV 的 P 正确性、DE 回归） |
| L0 | 六梯度（fp16 ×2 + fp32 ×5）——**主判据** |

### 预案树（按 L0 fp16 dq 的落点，四条分支各自的下一动作已定）
| dq 落点 | 含义 | 下一步 |
|---|---|---|
| **PASS（≤5e-3）** | D15+D13+DQ 三处全修好 | → fp16 其余五项逐个对；随后专攻 D7（fp32） |
| **0.05–0.2** | 主链通，残差 = dqkt 的 10%（B2 的 dP） | → 拆 B2：Vnew = U0 - W H^T 与 dO@Vnew^T，验 Vnew 的 4% |
| **≈1.0** | DQ 通道仍断 | → dump `dq1_l0` 与 `t1_ub`（L0C→UB 后的第一接点）；若 L0C→UB 不支持则改回 GM 但用全切片 |
| **nan / 507015** | `dq1_l0` 直拷引入的新问题 | → 回滚 `b47cbd06` 的 DQ 段，改用“全切片 + GM 中转” |

### 辅助判据（任一异常都能独立定方向）
- **S1G 的 A/H > 5e-3** ⇒ solve_a 除 D15 外仍有问题（A = (I+L)^{-1} 的三角前代）
- **S2AB 的 DGAM2 > 0.1** ⇒ dP 残差被放大（Vnew 错）
- **DBG 的 DQ absmax = 0** ⇒ 直拷未生效（回到预案树第三行）
- **DBG 的 Gam ≠ 2.5e-8** ⇒ D15 在 P/dA 段的推广引入了回归

## [40] D15 机理实证（生成码级）：列广播被折成标量 g[0]

### 对照实验（同一 kernel，f16，L=64 单 chunk）
| 版本 | 生成码关键段 | 上卡行为 |
|---|---|---|
| **基线 `9fc72a7b`（plain-2D）** | `Broadcast<axis=1>(row_tile,g_ub,{64,64},{64,1})` + `Broadcast<axis=0>(col_tile,g_ub,{64,64},{1,64})` + **`Adds(aux, row_tile, -col_tile.GetValue(0))`** | 能跑完；Gamma = `g_i − g_0`（D15 退化） |
| **plan-9**（GM→[1,C] + Broadcast + `g_ub[i]−aux[i,j]`） | `copy_gm_to_ub<float,64,1>(grow,S)` + `Broadcast<axis=0>` + **`Sub(dst, g_ub[outer], src, …, 1, …)` ×64 行** | **L0 七 case 全 507014**；scan2a 最小链 AICore 100% 空转（死等） |
| **plan-9b**（两片 2D 广播 + 纯 2D 相减） | `Broadcast<axis=0>(aux,g_ub)` + `Broadcast<axis=1>(dhp,g_ub)` + **`Sub<float>(aux,dhp,aux,…,64,…)`** | 待验（本轮在跑） |

### 结论
1. **D15 的真正机理**：`T.Parallel` 的 1D→2D 自动广播 lowering 会把**列项折叠成标量 `g[0]`**（不是“只算前 16 列”）。这与卡上实测（row10 全 0.0092、diag 0.6452）逐位吻合。
2. **plan-9 的挂死不是 CV flag 失配**：plan-9 与基线的 `CrossCoreSetFlag/WaitFlag` **序列逐项相同**（27 set / 27 wait，顺序一致）⇒ 问题在新增的 **UB 标量读** `Sub(dst, g_ub[outer], …)` 这条 codegen 路径。
3. **plan-9b 走的是编译器自己生成过的两条 Broadcast**（axis=0/1 都在基线上跑过），只把最后的合并从“标量折叠”换成两个 2D tile 相减 ⇒ 既修 D15 又避开标量读。

### 附：基线 UB 峰值 195936B（99.8% of 196352）——历史脆弱性的来源之一；plan-9b 的 alloc 与基线相同（复用 dhp 作行 tile，无新 buffer）。

### [40b] 第二个 D15 站点：solve_a（A 在 g≠0 下也错）
- `gdn_bwd_v2_solve_a` L241-245 仍是 plain-2D `gm_ub[i,j] = g_ub[i] - g_ub[j]` ⇒ 同一折叠 ⇒ A=(I+L)^{-1} 在 g≠0 下错
- 此前 S1G 报的 "A rel=5.27e-05" 是 **plan-1 双 serial 版** 的测量；`9fc72a7b` 回退后 A 又退化了（回退把三处都退回 plain-2D）
- **本轮已同修**：`T.tile.broadcast(gm_ub,g_ub,axis=0)` + `T.tile.broadcast(mul_ub,g_ub,axis=1)` + 2D 相减（`mul_ub` 稍后被 `fill(0)`，liveness 安全）
- scan2b 三处（L1261/1288/1323）是 `T.serial(i) + T.Parallel(j)` 形式：`g_ub[i]` 标量、`g_ub[j]` 连续向量读 ⇒ 不受该折叠影响，暂不动

## [41] plan-9b 步进判决：solve_a ✓、scan1 ✓、**scan2a 仍挂**

### 步进驱动（`_process/_w11p9_step.py`，逐 kernel synchronize）
| 步 | 结果 |
|---|---|
| 1 cumsum | **OK** |
| 2 **solve_a（plan-9b 两片广播）** | **OK** —— 向量-only kernel 里该形态可用；A 的 Gamma 首次以正确形态跑通 |
| 3 scan1 | **OK** |
| 4 **scan2a（plan-9b）** | **挂**（STEP 4 永不打印；AICore 100%，实例随后 SSH 掉线） |

### 关键对照
- 基线（plain-2D，退化但能跑）在 scan2a 里**本来就发过这两条 Broadcast**（axis=0/1）——说明 Broadcast 指令在 CV 混合核里可执行
- plan-9b 与基线的差异只剩：**行 tile 写进 `dhp_ub`**（该 buffer 之后既被 `copy_gm_to_ub(MskM)` 重写、又被 `Mul` 读取，并参与后续 AIC 侧 GM/L1 交接）
- ⇒ **最高置信假设**：把 row tile 放进 `dhp_ub` 引入了一条新的“AIV 写 → 后续 AIC 消费”的 buffer 级依赖，cv-sync pass 在此处生成/错排了跨核等待，形成死锁；**Broadcast 本身不是问题**（solve_a + 基线双重佐证）

### 下两车（实例恢复即发，均已定死）
1. **10b（首选，最小改动）**：给行 tile 一个**专用** `[C,C]` UB buffer（`rt_ub`，+16KB，预算 ~94%），不碰 `dhp_ub`；重跑步进驱动 → 若 scan2a OK 即证实上述假设
2. **10c（保底，结构性）**：把 Gamma 移到**向量-only kernel**（cumsum 里顺带算，或独立 `gdn_bwd_v2_gamma`），GM 落地 `WS_GAM[B,H,NT,C,C]`（真形 33.5 MiB）；scan2a/scan2b 只 `T.copy` 读取，彻底不在 CV 混合核里做列广播
3. 两车都可先用 `_w11p9_step.py`（步进）再跑 `_w11p9_one.py`（单 case）最后 L0；**不要**再用“日志静默”做 stall 判据（本轮误杀两次）

### 附带
- 误杀记录：`w11p9b2`/`w11p9one` 的本地轮询器按“日志静默”判死，实际在编译；后改为看 python 进程 CPU/缓存目录增长
- `solve_a` 的 `mul_ub` 作为行 tile：步进已证可用（mul_ub 随后被 `fill(0)`）

## [42] ★ 归因推翻：基线 scan2a 本身就在挂（新实例实测）

### 本轮对照（全部在 21:00 重建的新实例上）
| 车 | 内容 | 结果 |
|---|---|---|
| 10b | 专用 `rt_ub` 行 tile | scan2a 挂（生成码正确：两个 Broadcast + 纯 2D Sub，UB 183648/196352） |
| 10c | Gamma 移到 cumsum（向量-only），scan2a 只 GM 拷贝 | **仍挂**（生成码里已无 broadcast，只有 `copy_gm_to_ub(aux, WS_GAM)`） |
| 10d | 再把 M 从 `dhp_ub` 放回 `aux_ub`（基线车道） | **仍挂** |
| **CONTROL-B** | **基线 `9fc72a7b`（未改一字）走同一 wrapper** | **仍挂** |
| HEALTH | canary v0/v2（列/行广播）+ GEMM fp16 | **全 PASS（误差 0）** ⇒ 设备健康 |

### 结论
1. **“基线能跑”不再成立**：`9fc72a7b` 在新实例上同样卡死在 scan2a；此前“L0 能跑完”的记录来自更早的内核状态（dq1_l0 直拷/D13 之前），不是当前基线。
2. 因此本窗口对 plan-9 / 9b / 10b / 10c 的“挂死”判决**全部无效**——它们测到的是同一个既有死锁，与 Gamma 写法无关。
3. **死锁线索（生成码级）**：scan2a 的 AIV 在 chunk 循环中 `CrossCoreWaitFlag(1)`，而 AIC 在**程序末尾**才 `CrossCoreSetFlag<2,PIPE_FIX>(1)`；AIC 的第一句是 `CrossCoreWaitFlag(15)`，而 AIV 在等完 flag 1 之后才 set 15 ⇒ **wait-before-set 成环**。这是 AUTO_CV_SYNC 对 scan2a 生成的协议，与 Gamma 无关。

### 下一步（已定死，优先用现成工具）
1. 用窗十一留下的 `_process/_w11_scan2a_cut.py` + `_w11_gen_scan2a_cut.py` 做 **scan2a 源码截断二分**：逐段删 B1..B15，编译后 grep `CrossCoreWaitFlag(1)` 是否仍在 + 跑一步，定位是哪段引入这条反向等待
2. 重点候选：DQ 的 `dq1_l0`（AIC 末尾 `copy_l0c_to_gm(workspace_2, dq0_l0)`）+ `workspace_1/2` 两条 auto-GM 交接
3. 找到后：要么调整该段的交接写法（改回 GM 全切片中转），要么在上游 issue 里补一条“scan2a 形态触发 cv-sync wait-before-set”

## [43] ★★ 死锁根因确认并修复：DQ 的 L0C→UB 与 E 段 spill 复用同一 auto-GM buffer

### 二分判决（当前源码截断探针，`_w11_gen_scan2a_cut.py` 重新生成）
| cut | 内容 | 结果 |
|---|---|---|
| 6 | 全链 | **timeout（挂）**；静态：`wait1_hits=6`、`ws12_l0c_copies=6` |
| 5 | 去掉 F 段（`# ---- dW -> WS_DW` 起） | **PASS**；静态：`wait1_hits=6`、`ws12_l0c_copies=2` |

### 生成码证据（cut6）
```
270 AIC: copy_l0c_to_gm(workspace_1, dq1_l0, …, 64, 0, 0)
281 AIC: copy_l0c_to_gm(workspace_2, dq0_l0, …, 64, 0, 0)
282 AIC: CrossCoreSetFlag<2,PIPE_FIX>(1)
342 AIV: CrossCoreWaitFlag(1)          ← E 段（chunk 循环内）
346 AIV: copy_gm_to_ub(xk_ub, workspace_2, …)   ← E 段的 xk_ub spill 回填
```
AIC 的第一句是 `CrossCoreWaitFlag(15)`（由 AIV 在等完 1 之后才发）⇒ **wait-before-set 成环**。
根因：内存规划把 **E 段的 UB spill buffer** 与 **F 段 dq0 的 L0C→UB 交接 buffer** 复用成同一个 `workspace_2`；cv-sync 按 buffer 身份建跨核边，产生不可满足的反向等待。**与 Gamma/broadcast 无关**。

### 修复（已上卡验证）
给 DQ 的两条 L0C 交接分配专用显式 staging（`WS_DQ1`/`WS_DQ0`，`[B*H,C,C]` fp32，普通输入、不进 workspace_idx）：
```
T.copy(dq1_l0, WS_DQ1[cid,:,:]); T.copy(WS_DQ1[cid,:,:], t1_ub)     # 934/940
T.copy(dq0_l0, WS_DQ0[cid,:,:]); T.copy(WS_DQ0[cid,:,:], xk_ub)     # 963
```
步进判决：`STEP 1 cumsum OK / 2 solve_a OK / 3 scan1 OK / 4 scan2a OK / 5 scan2b OK / STEP-DONE` —— **首次全链无挂死**。

### 单 case（L0 basic fp16）数值
| 梯度 | rel | 判 |
|---|---|---|
| dq | **3.909** | FAIL（非 nan，与 D13 前同一量级） |
| dk / dv / dbeta / dg | **nan** | FAIL（scan2b 侧） |
| dh0 | 0.848 | FAIL |

⇒ 死锁层关闭，回到**数值层**：dq 的 3.909（scan2a B4 链）与 scan2b 的 nan 是下一步两条独立线。

## [44] 现场归档（实例到期前拉回，artifacts/w11ev/，1.0 MB）

### 截断补全
| cut | 内容 | 结果 |
|---|---|---|
| 6 | A+B+C+D+E+F（全链） | **挂（timeout）** |
| 5 | A+B+C+D+E | **PASS** |
| 4 | A+B+C+D | **PASS** |
| 3 | A+B+C（跳过 P/Gamma 之后的一切） | FAIL（TVMError：gating 删掉了后续段依赖的定义，编译期，符合预期） |
⇒ **挂死由 F 段单独引入**（与 E 及之前无关）。

### 归档内容（`artifacts/w11ev/`）
- `w11p10e/*/kernel.cu` ×5：最终版五核生成码（cumsum 89L、solve_a 181L、scan1 351L、**scan2a 778L**、**scan2b 817L**）
- `w11one.log`：单 case 六梯度判决（dq 3.909 / dk,dv,dbeta,dg nan / dh0 0.848）
- `w11step.log`：步进五步全 OK（死锁修复凭据）
- `w11cut3_*.log` / `w11cut2_all.log`：cut 6/5/4/3 判决

### 下窗开局（实例恢复后，全部离线可做）
1. **scan2b nan**：直接读 `w11ev/w11p10e/*/kernel.cu`（817L）里的 scan2b 生成码，先静态找 nan 源（exp/除零/未初始化 UB）；再决定是否要 `_w11_num_diag.py`（需按新签名更新：cumsum 双输出、scan2a 多两个 staging）
2. **dq 3.909**：同法读 778L 的 scan2a 生成码，重点 B4 段（dqkt→da_l1→dq1_l0→WS_DQ1）

## [45] DQ=0 的根因（生成码级）：显式 staging 不在 workspace_idx ⇒ cv-sync 不建跨核边

### 判定
- `ITEM dq rel=3.909` 恰等于 golden dq 的 absmax（3.909）⇒ **卡上 DQ 全 0**（非数值误差）
- 生成码（`w11ev/w11p10e/b6e3…/kernel.cu`）：
  - AIC 270：`copy_l0c_to_gm(WS_DQ1, dq1_l0, 64,64,64,0)` —— **其后无任何 `CrossCoreSetFlag`**
  - AIC 280：`copy_l0c_to_gm(WS_DQ0, dq0_l0, …)`（唯一的 set 12 在这句**之前**）
  - AIV 736：`copy_gm_to_ub(t1_ub, WS_DQ1, …)` —— **前面没有 `CrossCoreWaitFlag`** ⇒ 读在写之前，读到未初始化 GM（全 0）
- 对照：`workspace_ldk`（在 `workspace_idx` 内）的 L0C→GM→AIV 读被正确配了 flag 11/12
⇒ **cv-sync 只对 `workspace_idx` 里的张量建跨核边**；普通输入张量（WS_DQ1/0）即使跨核读写也不建

### 修复（已落本地，待上卡）
把两个 staging 改名为 `workspace_dq1/workspace_dq0` 并加入 `workspace_idx=[10,20..28,30,31]`（与 `workspace_ldk` 同类），由 runtime 分配；wrapper/驱动去掉显式 `torch.empty` 与实参。

### 上卡验证清单（实例恢复即跑）
1. `_w11p9_step.py`：五步全 OK
2. `_w11p9_one.py float16`：看 dq 是否从 0 变为有限且 ratio 上升
3. `w11nan_fire.sh`（ND/BOX/ZERO 三腿）：定位 scan2b nan

## [46] 数值层三车判决（新实例 22:30）：deadlock 修复后逐项收敛

### 已修复并上卡验证
| 修复 | 证据（上卡） |
|---|---|
| `workspace_dq1/0` 纳入 `workspace_idx`（跨核 flag 建边） | **DQ 从全 0 → out0 rel=3.148e-02** |
| `workspace_ldk/lbd` 改 4-D `[B*H,bk_num,C,BK]`、半块用前置索引 `[cid,hh,:,:]`（realTailN 由 1 → 64） | **DH0 0.8478→0.0899（ratio 0→0.0188）**、**WS_DK 1.0→3.68e-02** |

### scan2b 清白（ZERO 腿）
七个边界 box 全零 ⇒ 四输出（dkg/dvg/dbeta/dg）**全 0、无 nan** ⇒ scan2b 自身干净，nan 全由 box 带入。

### 当前 box 表（h0=on，9 项）
| box | rel | 判 |
|---|---|---|
| out0 DQ | **3.15e-02** | 大幅改善（原全 0） |
| out1 DH0 | 4.54e-01（alt dvnew 形式 1.06e-01） | 有改善仍错；B6b 形式待定 |
| out2 WS_DV | 1.00 | 错（有限） |
| out3 WS_DW | 3.98，**nan=3456** | ✗ 唯一 nan 源（dW 路径） |
| out4 WS_DKT | 3.8e30（golden=0） | ✗ 垃圾 |
| out5 WS_DE | 1.98e-04 | ✓ |
| out6 WS_DGAM2 | 3.77e-02 | 基本对 |
| out7 WS_DGAM | 3.16e17（golden=0） | ✗ 垃圾 |
| out8 WS_DK | **3.68e-02** | ✓ |

### 尝试与回退
- **slot 去别名**（把 ldk 按 W/dW/dK/dKt 拆 4 槽）：**回归**（dv 0.485→nan），已回退，保留 4-D 布局。
### 下一步两条线
1. **WS_DW nan**：dW 路径 = `gemm_v0(bq_l1, hin_l1, dw0_l0)`，其中 `bq_l1` 来自 `workspace_vcb`；vcb 同时承载 beK（前）与 dVnew（后）——同一类跨时段复用/时序问题，需查 vcb 的写读时序或单独开 staging。
2. **WS_DKT / WS_DGAM 垃圾（golden=0）**：查 `dkt0_l0` / `dgam_ub` 是否真被写；h0=zero 时 out4=DKT 与 out8=DK 同值（3.807）提示两者可能写同一源。

## [47] 未初始化读两处（已修）+ 当前数值全景

### 已修并上卡验证（`fe125fd6`）
| 问题 | 机理 | 上卡证据 |
|---|---|---|
| **WS_D（Delta' relay）首块读未初始化** | auto workspace 由 runtime `empty` 分配；最深块（`cc==chunk_num-1`）本应 Δ'=0，却从 GM 读入垃圾 → 污染 dVnew / ⟨H_in,Δ'⟩ / dgam / dW | 两处 Δ' 读取改为该块跳过（保留前置 fill 0） |
| **dKt 读未初始化 L0C** | `dkt0/1_l0` **没有任何 GEMM 写**（`vnew @ Δ'^T` 未实现），却被拷入 `WS_DKT` | **WS_DKT rel 3.8e30 → 0.000e+00**、**WS_DGAM 3.2e17 → 0.000e+00** |

### 当前数值（L0 basic fp16，wrapper 路径）
| 梯度 | rel | ratio | 判 |
|---|---|---|---|
| **dq** | **1.940e-03** | **0.9346** | 接近过（上限 1e-1，但 required ratio 0.99） |
| dk | nan | 0 | 由 WS_DW nan 带入 |
| dv | 4.847e-01 | 0.0027 | 错 |
| dbeta / dg | nan | 0 | 同上 |
| dh0 | 8.989e-02 | 0.0188 | 有改善仍错（alt dvnew 形式 1.06e-01） |

### 尝试并回退
- `vb2/vn2` 拆分（dO / beta·V / Vnew 三次复用 vcb 的 WAR 隐患）：**使 WS_DV 由有限变 nan**（3456 个），且 dq 无变化 ⇒ 已回退（`workspace_idx` 同步还原）。

### 唯一剩余 nan 源：`WS_DW`
- 三个 case 均 `WS_DW nan=3456`（54 行×64），来源在 dW 路径：`gemm_v0(bq_l1, hin_l1, dw0_l0)` → `workspace_ldk` → AIV。
- 已知线索：dW 的 L0C→GM 现在已是 64 列（4-D 布局修复）；dVnew（out2）有限但 rel=1.0；`WS_DW` 与 `WS_DV` 的 nan 计数恒等（3456），提示二者写回共用同一中间 lane/同一时刻的坏缓冲。
- 下一步：在 dW 写回前把 `xk_ub`（fp32，转 fp16 之前）借 `WS_DGT` 槽 dump 一次，区分「dw0_l0 本身含 nan」vs「AIV 侧读/转写出问题」。

## [48] dW lane dump：fp32 有限但 ≈0 ⇒ nan 不在 dw0_l0，dVnew≈0

- 新开关 `GDN_S2A_DW_DBG=1`（默认关）把 **fp32 dW lane**（fp16 写回之前）借 DKT 槽 dump；dqkt dump 让位
- 上卡：`DUMP-dWlane rel=1.000e+00 nonfinite=0 gold_max=7.206e-01` ⇒ **dW≈0**（不是 nan）
- 推论：`WS_DW` 的 nan 来自写回路径之外的别处；且 dW = dVnew·H_in ≈ 0 说明 **dVnew≈0**（与 WS_DV rel=1.0 吻合）
- 新嫌疑：`workspace_lcb` 在同一次 BV 迭代里既承载 `u0`（AIC 读）又承载 `dvnew`（AIV 写）——与本次已修的 ldk 同类 1W1R 复用；P 的交接（vcc，flag 22）已核对有序
- 下一步：把 dvnew 的交接迁到专用 workspace（或在写回前 dump `dvnew_l0` 本身），确认 dVnew 是"算错"还是"写丢"

## [49] 第三次 de-alias 负结果：dvnew 专用 workspace 把 dv 从 0.485 打成 nan

- 改动：`workspace_ldv`（专用，[B*H,C,BV] fp32），dvnew 两处交接从 `lcb` 迁出
- 上卡：**dv 0.4847 → nan**，其余不变（dh0 0.0899）
- 与 [46] 的 slot 拆分、[47] 的 vb2/vn2 拆分**同型**：任何给交接缓冲"开专用槽"的改动都会让 UB/L1 liveness 变化，进而改变 codegen 时序，产生新 nan
- **结论修正**：`WS_DW ≈ 0 / WS_DV` 的病灶不能靠"消除别名"来修；别名只是表象，真正的问题是某些跨核交接**本身缺少时序保证**（或依赖了别名带来的巧合顺序）
- 当前回退到稳定态（`3f223157`）：dq 1.94e-3 / dh0 0.0899 / dv 0.485 / dk·dbeta·dg nan（由 WS_DW 带入）
- 下窗方向（需换思路）：用 `GDN_S2A_DW_DBG` 已证明 dW lane 有限≈0，下一步直接 dump **dvnew_l0 的 L0C 值**（写回之前）与 **p_l1 指针处的值**，确认 dVnew 的 GEMM 到底有没有算；若算对了则病灶在"AIC 写 L0C → AIV 读"的跨核可见性，需查该链的 flag 配对

## [50] ★ dVnew 写回前即为 ≈0（有限）——病灶在 GEMM 输入侧

- `GDN_S2A_DW_DBG` 改为 dump **写回前的 fp32 dVnew**（原 dW dump 撤下）
- 上卡：`DUMP-dVnew rel=1.000e+00 nonfinite=0 gold_max=7.267e-01` ⇒ **dVnew≈0**（有限）
- 因此既不是 fp16 转换，也不是 WS_DV 写回；问题在 `gemm_v0(p_l1, bq_l1, dvnew_l0)` 的两个输入
- **关键对照**：dGam（out6=3.77e-2）用的是 **`pk2_ub`（UB keep-alive）里的 raw**，而 P 用的是 **`workspace_lcc`（GM stash）里的 raw**。dGam 对、P 路为 0 ⇒ 高度指向 **`workspace_lcc` 的读（AIV 侧 wait 10 之后的 GM→UB）拿到的是 0/未就绪数据**
- 下窗一击：在同一 debug 开关下改 dump `aux_ub`（即从 lcc 读出的 raw，乘 Gam 之前），若为 0 则实锤 lcc 读失效；否则查 bq_l1（dO）与 p_l1 的 L1 装载

## [51] ★ 假象修正：dVnew 真值为「3456 个 nan」——唯一共同源头

### 假象来源
- `if not _S2A_DBG:` 的 **dKt 零写**在 `_S2A_DW_DBG=1` 时依然执行，把 debug 槽（WS_DKT）在图末尾覆盖成 0
- 因此 [50] 的 "dW≈0" 与 "P 段 Gam=0" **均为假象**；已在 `fe…` 之后修正为 `if not _S2A_DBG and not _S2A_DW_DBG:`

### 修正后的真判决
```
DUMP-dVnew rel=6.035e+00 nonfinite=3456 gold_max=7.267e-01
```
- **dVnew 本身含 3456 个 nan**（与 WS_DV、WS_DW 的 nan 计数完全一致）
- 因果闭环：`dVnew` nan → DV=beta·t1 nan、dW=dVnew·H_in nan、dA nan → dbeta/dk/dg nan
- 已排除：raw（因 dGam 用同一 lane 且正确）、Gam（cumsum 单独验证：diag 全 1、row10 与 golden 一致、absmax 1.6e11 有限）、M（源码顺序核对）

### 下窗一击
在 BV 循环的 GEMM 之前 dump 两处：
1. `io_cc`（fp16 的 P，经 aux_ub 转 fp32 后写 WS_DKT）——看 P 是否已含 inf/nan
2. `bq_ub`（fp16 的 dO）——同上
判读：P 含 inf ⇒ 上三角未被 M 清零（fp16 溢出）；dO 含 nan ⇒ vcb 交接读到坏数据。

## [52] 复盘（2026-10-18 晚场，实例已停）：约一半卡时是无效消耗

### 时间线与逐车判决（B 窗 22:29–23:37，新实例）
| 时刻 | 车 | 目的 | 结果 | 有效? |
|---|---|---|---|---|
| 22:33 | w11all#1 | step/one/zero/box/nd 五腿 | step OK；**ZERO 证 scan2b 清白**；box 定位 DW/DKT/DGAM/DK | ✅ |
| 22:39 | w11all#2 | ldk/lbd 改 4-D | dh0 0.848→0.0899、DK 1.0→3.7e-2 | ✅ |
| 22:44 | w11all#3 | **slot 去别名** | dv 0.485→nan | ❌ 回归 |
| 22:48 | w11all#4 | Δ′ 首块跳过 + dKt 零化 | DKT/DGAM → 精确 0 | ✅ |
| 22:53 | w11all#5 | **vb2/vn2 去别名** | WS_DV 有限→nan | ❌ 回归 |
| 22:57 | w11fin | 回退态复核 | dq 1.94e-3 等 | ✅ |
| 23:13 | dW dump | 判 dw0_l0 是否有 nan | “dW≈0” | ❌ **假象** |
| 23:17 | ldv 去别名 | 拆 dvnew 交接 | dv→nan | ❌ 回归（第三次同类） |
| 23:21 | dvnew dump | 判 dVnew | “dVnew≈0” | ❌ **假象** |
| 23:29 | Gam dump | 判 P 段 Gamma | “Gam=0” | ❌ **假象**（diag 全 0 本身就违反不变量） |
| 23:31 | dvnew dump（修正覆盖后） | 复测 | **dVnew nan=3456** | ✅（但探针本身扰动 liveness） |
| 23:35 | P dump | 判 fp16 P | **上传 md5 不符，跑的是旧码** | ❌ 浪费 |

### 无效消耗统计（B 窗约 95 分钟）
- 三次 de-alias 失败：~15 分钟（外加三轮代码 churn）
- 三次假象 dump + 一次重复：~20 分钟
- 陈旧截断探针（cut6 用旧快照）：~5 分钟
- `_S2A_DBG` 未定义导致整轮编译失败：~5 分钟
- 上传未校验就发车（最后一次）：~4 分钟
- 两车并发后杀车/重发：~3 分钟
- **合计 ~50 分钟，占 B 窗一半以上**

### 四个系统性原因（按代价排序）
1. **向共享调试槽写 dump，没有枚举该槽的全部写点**。`WS_DKT` 上已有 `if not _S2A_DBG:` 的 dKt 零写，在新开关下依然执行，把 dump 覆盖成 0。一次 `grep WS_DKT[` 即可发现。三条假象判决全部由此产生。
2. **同一类被否的假设重复三次**（de-alias）。第一次回归后就应判定“该类做法无效”，而不是换一种粒度再试。
3. **探针无自检不变量**。“Gam 全 0 且 diag=0” 在数学上不可能（diag 必须为 1），当时的正确结论是“探针坏了”，我却当成“kernel 坏了”。
4. **发车前不校验输入**。最后一次 P dump 上传 md5 不符仍继续发车，跑的是上一版源码；另有并发两车争卡。

### 现场整改（可执行，已写入下窗流程）
- **探针四律**：① 新 dump 目标必须是**专用槽**，若复用已有张量先 `grep` 出全部写点并列出条件；② 每个探针必须声明**不变量**（Gam diag=1、gold max>0、count 与已知 nan 数一致），先验不变量再解读；③ 探针本身会扰动 liveness，**结论需第二条独立路径**（例如用 dump 出的 P 与 dO 在 host 重算 dVnew）交叉验证；④ 同一类做法失败一次即登记为“已否”，重试必须给出新的因果理由。
- **发车三律**：① 一车一锁（卡上 `flock`），禁并发；② 发车前打印**全量 md5 清单**并在卡上比对，不符即拒跑；③ 每车必须先在本地过 `--golden-only` 或 `py_compile` + 关键 grep。
- **账目纪律**：一条判决写进台账时必须同时写“支撑它的唯一测量点”；出现假象后不删旧条目，追加勘误（本文件已按此处理：[48]/[50] 的“≈0”结论由 [51]/本节的“假象”更正）。

### 仍然成立的收益（本晚净得）
| 项 | 提交 | 凭据 |
|---|---|---|
| 死锁根因与修复（spill 与 handoff 复用同一 auto-GM 槽 → cv-sync wait-before-set） | `adf0c794` | 步进五步全 OK |
| DQ staging 入 workspace_idx | `08a9c5eb` | DQ 全 0 → 1.94e-3 |
| 半块 workspace 4-D（realTailN 1→64） | `29173bfc` | DH0 0.848→0.0899、DK 1.0→3.7e-2 |
| Δ′ 首块零化 + dKt 零化 | `fe125fd6` | DKT/DGAM → 精确 0 |
| scan2b 清白（零输入对照） | 工具 | 四输出全 0 无 nan |

### 下窗唯一实验（含前置校验）
1. 本地：给探针加不变量断言；`grep WS_DKT[` 列出全部写点并确认互斥
2. 发车前：md5 清单比对 + 单锁
3. 实验：dump **fp16 的 P（io_cc）** 与 **dO（bq_ub）** 各一次（分两腿，避免同槽覆盖），判读：
   - P 含 inf ⇒ 上三角未被 M 清零（P 组装顺序或 M 读取问题）
   - 两者干净 ⇒ 用 host 重算 `P^T dO` 与 dump 的 dVnew 对比，判定 GEMM 本身

## [53] 全面 review（实例已停，纯本地）：断言面 / 结构 / 调用方 / 证据 / 验证缺口

> review 不是"看哪里不对"，而是四类可枚举对象的逐条取证。

### 一、断言面（约束注释 / 命名 / 索引 / 台账数字）
| # | 断言 | 检验结果 | 处置 |
|---|---|---|---|
| A1 | `workspace_idx` 12 个索引 | 机器解析：全部指向 `workspace_*`，无越界 | ✅ |
| A2 | "`_S2A_DW_DBG` = dump the fp32 dW lane" | 实现已改为 **P dump** | ❌ 伪断言 → 注释已重写 |
| A3 | "`_S2A_DBG` = dqkt instead of dKt" | dKt 现已恒 0（未实现） | ⚠ 半伪 → 注释已重写 |
| A4 | `dkt0_l0/dkt1_l0 = "dKt half accum (B9)"` | **无任何读写**，纯死结构 | ❌ 已删 alloc（两个 L0C 预算退回） |
| A5 | dKt 处 "not implemented … emit zeros" | 与实现一致 | ✅ |
| A6 | `workspace_dq1/0` 注释 | **缺"必须在 workspace_idx"这条最贵的约束** | ❌ 缺失断言 → 已在声明处写明原因（否则会被改回普通参数、DQ 再次全 0） |
| A7 | 台账 [48]/[50] 的 "dW≈0 / dVnew≈0 / Gam=0" | 三条均为 dump 被 dKt 零写覆盖的**假象** | ❌ 已在 [51]/[52] 更正；提交 `f57f92e0` 的 message 与事实不符，git 不可回写，此处登记 |

### 二、结构/边界
- 死结构：`dkt0/1_l0`（已删）。删除后 L0C 静态量回到 11 个 alloc（实际峰值不变，liveness 复用）。
- 命名即契约：`_S2A_DW_DBG` 现在语义是"第二调试槽（P dump）"，与名字不符。**暂不改名**（会破坏既有脚本与台账引用），以声明处注释为准。
- 同名不同形：`workspace_ldk/lbd` 在 scan1 是 3-D、scan2a/2b 是 4-D —— 有意（不同 kernel 各自声明），注释已分别标注 "half blocks"。

### 三、调用方扩散（本次签名变更的真实代价）
`gdn_bwd_v2_cumsum` 变双输出、`scan2a` 增 `WS_GAM` 后，**6 个探针工具静默失效**（探针不进测试，无人报错）：
| 文件 | 状态 | 处置 |
|---|---|---|
| `_w11_s2a_dbg.py` | 1 参数解包 + 10 实参 | ✅ 已修为双输出 + 11 实参 |
| `_w11_scan1_g.py` | 1 参数解包 | ✅ 已修（`ws_g, _ws_gam`） |
| `_w11_gen_scan2a_cut.py` | 生成的 driver 与 kernel 缺 ws_gam / 缺 `_S2A_DW_DBG` | ✅ 生成器已自带两处，**重新生成并校验通过** |
| `_w11_scan1_cut*.py` / `_w11_scan1_micro.py` / `_s7f_stage_*.py` | 陈旧 | 历史件，不再使用（未改） |
| `_w11p9_step.py` / `_w11_num_diag.py` / `_w11_s2a_boxes.py` / `_w11_s2b_zero.py` | 已同步 | ✅ |

### 四、证据可复核性（最严重的流程债）
- B 窗（22:29–23:37）**所有车日志都在卡上**，实例已停 → 丢失；只有轮询片段在会话里。[46]–[52] 的数字**无法再从原始日志复跑**。
- 已落地的对策：`scripts/w11_car.sh` —— 清单 md5 校验 + `flock` 单锁 + 活性轮询（看进程/缓存，不看日志静默）+ 判决自动 pull 到 `artifacts/w11car/`。
- 旧证据仍在：`artifacts/w11ev/`（五核生成码 + 旧日志，对应 [43] 的数字）。

### 五、验证缺口（当前结论的适用范围）
- 本轮改动后**只跑过 fp16 basic（`one`）与 h0=on 的 `box`**；
- **fp32 与 typical（L=256）自改动后从未跑**；L0 全量自 19:0x 起未跑。
- ⇒ "dq 1.94e-3 / dh0 0.0899 / dv 0.485" **仅对 L0 basic fp16 成立**，不得外推。

### 六、下窗唯一实验（带防重犯前置）
1. `grep -n "WS_DKT\[" gdn_bwd_v2.py` 确认为单写者（dqkt dump 与 dKt 零写均对 `_S2A_DW_DBG` 让位）
2. `bash scripts/w11_car.sh "dw"`（清单校验 + 单锁 + 自动归档）
3. 判读：`PROBE-BROKEN-all-zero` ⇒ 探针坏（先修探针，不下结论）；否则看 `DUMP-P nan/inf/upper_absmax`
4. 无论结果，**必须用 dump 出的 P 与 dO 在 host 重算 `P^T dO`** 与 dVnew 对照（探针本身扰动 liveness，单点读数不算证据）

## [54] 十余窗口复盘（决策 + 代码），文档见 `release/ascend/port910b/review_gdn_bwd_v2_windows.md`

一句话结论：**结构决策（五核/CV 交接走 GM/半块）成立；决策过程有五类系统性缺陷，代价是七个窗口的返工与今夜一半卡时。**

五类缺陷：①用文档承诺替代能力探针（五个平台语义探针合计 <50 行，却能省掉 D7/D13/D15/死锁/workspace_idx 五条链）；②验证输入退化（g≡0/单 chunk/h0=0 成为缺陷掩体）；③症状直接跳机制，绕过生成码这一唯一事实源；④缺 baseline 对照就归因（四个候选修复的挂死判决作废）；⑤修复方向由"代码更干净"的直觉主导，而真实约束是 cv-sync 归属与预算悬崖；另加⑥探针扰动被测对象、⑦证据不入库。

已落地的整改：design.md §18（平台约束/未实现项/已否清单）、`scripts/w11_car.sh`（清单校验+单锁+自动归档）、探针四律与发车三律、死结构与伪注释清理、6 个失效驱动修复。

## [55] 本地准备完成（实例停）：下窗一车即可出 dVnew 判决

### 已落地的四件
1. **`_S2A_DW_DBG` 升级为取值开关**（默认 0）：`1` = dump fp16 P（cast 回 fp32），`2` = dump fp32 dO（`bq_ub`）。任一非零都让 dKt 零写与 dqkt dump 让位，WS_DKT 单写者不变。
2. **`_w11_s2a_boxes.py`**：每次运行保存九个输出到 `/tmp/w11box_out{i}.npy`；DUMP 模式另存 `/tmp/w11dump_{mode}.npy`，并打印不变量自检 `PROBE-OK/BROKEN-all-zero`。
3. **`_w11_recompute_pv.py`**（本地，无需 NPU）：读两份 dump，host 重算 `Pᵀ dO`，与 golden、卡上 `WS_DV` 三方对照，直接给 `VERDICT operands-OK/operands-BAD`。无 dump 时优雅退出（已实测）。
4. **`_w11p9_one.py` 支持任意 L0 档**：`basic|basic32|typical|h0|tail`（typical = H=2、4 chunk，同时验 cid 偏移与多 chunk）。
5. `scripts/w11_car.sh` 新增腿：`dw2`、`typ`、`h0`、`one32`、`gate`，并在 [4] 自动 pull 三个 `.npy`。

### 下窗两车（命令与判读已定死）
```bash
# 车 1：dVnew 定位（约 25 min）
bash scripts/w11_car.sh "dw dw2 box zero"
python3 <local>/_w11_recompute_pv.py        # 读 artifacts/w11car/*.npy
# 车 2：覆盖缺口（约 35 min）
bash scripts/w11_car.sh "typ gate nd"
```
判读：`VERDICT operands-OK` ⇒ 换 GEMM 形态（dO 转置后走 tB，GP5 已验证）；`operands-BAD` ⇒ 按 P/dO 哪个非有限回到对应交接链。车 2 的 `typ` 同时回答"`cid` 偏移对不对"这一可能更致命的问题。

## [55] openspec apply：回写计划（G1–G6）+ 本地就绪任务 G5 关闭

### 计划回写（R10 缺口修补）
`p2-13-ascend-runtime/tasks.md` 新增 **§8 gdn_bwd_v2 数值收敛**，把"下一步计划"落成六个可勾选孙任务（G1 定位 / G2 旁证 / G3 分支修复 / G4 覆盖缺口 / G5 输入门 / G6 dKt gap 决策），验收锚随任务行前置。

### 本轮完成的本地任务
| 任务 | 产物 | 凭据 |
|---|---|---|
| **G5 三档输入门**（已勾选） | `_w11_gate.py` + `w11_car.sh` 前置（fail-closed） | `GATE-PASS`，矩阵显示 7 case 覆盖 g≠0/多 chunk/h0≠0/fp32 全 True；`l0_special` 显式标 `nan=1` |
| **G1/G2 就绪**（待上卡） | kernel mode 开关（P/dO 单写者）+ 驱动 npy 落盘 + host `PᵀdO` 三方对拍 + `PROBE-PV-SKIP` 分支 | 本地 `py_compile` 全过；`WS_DKT` 三模式单写者已 grep 证实；车腿 `dw`/`dw2` 就绪 |
| **G3 车腿**（部分） | `onet`/`one32`/`h0` 腿修正（此前把 case 传进 dtype 位，四条腿会 KeyError） | `bash -n` OK；`_w11p9_one.py` 支持 dtype+case 且返回真实退出码 |

### R30 独立偏差评估（CodeReview 子代理）与处置
| 级别 | 发现 | 处置 |
|---|---|---|
| P0 | 四条车腿把 case 名传进 dtype 位（必 KeyError） | 已修：`one/onet/one32/h0` 全部 `dtype + case` |
| P1 | 前置 gate 走管道判码 ⇒ 恒 exit 0（fail-open） | 已改：捕获输出再判，`ABORT input gate` |
| P1 | `PV-recompute` 判据行被两处 grep 过滤掉 | 已修：grep 增加 `PV-`，归档保留 40 行 |
| P2 | dump 模式下 `out4_WS_DKT` 仍按"应为 0"对拍 → 假误差 | 已修：dump 模式跳过并标注 |
| P2 | `/tmp` 旧 npy 跨窗残留；缺失时静默 | 已修：发车清 `w11dump_*.npy`；缺 P 时打 `PROBE-PV-SKIP` |
| P2 | gate 对 nan 输入判 False（`l0_special`） | 已修：`nanmax` + 单独 nan 计数 |
| P2 | 轮询上限 30 min < 腿预算；早拉无提示 | 已修：80 min + `PULL-EARLY` 提示 |
| P2 | G1 任务文本写"同跑/WS_DGAM2"，实现是两次 run/同箱 | 已修：文本改成两腿两 run |

## [56] ★ G1/G2 判决 → D16 双重 exp + dVnew 交接竞争，两处修复见效

### G1/G2 车（`dw`/`dw2`/`zero`，10-21 14:23–14:26）
```
DUMP-P  absmax=nan nan=1397 inf=0   PROBE-OK
DUMP-dO absmax=4.113e+00 nan=0 inf=0 PROBE-OK
PV-recompute vs golden rel=6.035e+00 | card WS_DV vs golden rel=1.000e+00
```
离线解剖归档 npy（零卡时）：
- NaN **全部在上三角（1397/2016）**，下三角/对角零 NaN；有限上三角全为 0（掩码已生效）
- `gam_inv = P/raw` 反解：**对角 = 2.717 ≈ e**，且逐项 `exp(golden_gam)`（0.6452→1.9064、1.0→2.718、0.4969→1.6436）
- ⇒ **D16：P 段的 Gamma 被做了两次 exp**（plan-10c 换成 GM 读后残留 `T.exp`）
- ⇒ 上三角 `exp(1.6e11)` → +inf，inf×M(0) → **NaN**，与 1397 个位置完全对应
- ⇒ dA 段读同一 GAM 但无第二次 exp，故 dq 正确而 P/dVnew/dW 错

### 修复 1（D16）：删 P 段多余的 `T.exp`
box 结果：`out3_WS_DW 3.977+3456nan → 4.756e-04`（零 nonfinite），其余箱不变。

### 修复 2：dVnew 的 GM 中转竞争
- 现象：`out2_WS_DV rel=1.000`（card 全 0），但 `dW` 用**同一个 L0C 值**却已正确 ⇒ GEMM 对、写回断
- 机制：`dvnew_l0 → workspace_lcb(GM) → t1_ub → io_bq → WS_DV`；lcb 每轮被写三次（dvnew/u0/qh），**第一次 AIV 读与 AIC 写竞争**（同 [45] 同类）
- 修复：**L0C→UB 直拷**（`T.copy(dvnew_l0, t1_ub)`，dq1_l0 先例）+ 经空闲 `io_cc` 做 fp32→fp16（bisheng 不接受混合 dtype 的 UB→GM 直写）
- 结果：**`out2_WS_DV rel 1.000 → 6.785e-04`**

### 当前 box 全景（h0=on，scan2a 九箱）
| 箱 | rel | nonfinite | 判定 |
|---|---|---|---|
| DQ | 3.148e-02 | 0 | 待改善（目标 <2e-2） |
| DH0 | 4.536e-01（alt 形式 1.061e-01） | 0 | 待查 |
| **WS_DV** | **6.785e-04** | 0 | ✅ |
| **WS_DW** | **4.756e-04** | 0 | ✅ |
| WS_DKT | 0 | 0 | 未实现（预期） |
| WS_DE | 1.978e-04 | 0 | ✅ |
| WS_DGAM2 | 3.772e-02 | 0 | ✅ |
| WS_DGAM | 0 | 0 | ✅（golden=0） |
| WS_DK | 3.680e-02 | 0 | ✅ |
| zero 腿 | 全 0 | 0 | scan2b 清白 |

### 仍待解（`one` basic 全链）
`dq 1.940e-3 / dk·dv·dbeta·dg = nan / dh0 8.989e-2` —— scan2b 的 nan 不由上述九箱解释（九个输入均有限），下一车 `nd` 定位首现箱。

## [57] D16 + DV 写回两修之后：scan2a 八箱全绿，病灶收窄到 scan2b

### basic（h0=zero）九箱（`boxb`，10-21 15:15）
| 箱 | rel | 判定 |
|---|---|---|
| DQ | **4.918e-04**（原 3.148e-02） | ✅ |
| DH0 | 4.581e-01（alt dvnew 形式 1.060e-01） | ❌ 待查 |
| WS_DV | 6.785e-04 | ✅ |
| WS_DW | 0（golden=0） | ✅ |
| WS_DKT | 0 | 未实现（预期） |
| WS_DE | 0（golden=0） | ✅ |
| WS_DGAM2 | 5.212e-04 | ✅ |
| WS_DGAM | 0 | ✅ |
| WS_DK | 4.721e-04 | ✅ |

h0=on 九箱（`box`）同样：DV 6.785e-04 / DW 4.756e-04 / DE 1.978e-04 / DGAM2 3.772e-02 / DK 3.680e-02 / DKT 0；DH0 4.536e-01。

### one（basic 全链）——nan 已消灭，剩有限错值
```
dq 1.940e-03 (0.9346)   dk 6.733e+00   dv 1.601e+00
dbeta 1.526e+01         dg 2.740e+00   dh0 8.989e-02
```
- 修复前：dk/dv/dbeta/dg = **nan**；修复后全部有限 ⇒ D16 的 NaN 链已断
- scan2a 输入（九箱）在 basic 下全部 ≤6.8e-4 ⇒ **病灶在 scan2b 自身**
- `nd` 车（直接调 scan2a/scan2b）同 case 全链 nan=0/inf=0，与之一致

### scan2b 的性质与下一步
- scan2b 是 **v1 遗留实现**（窗十之前），`P1-1w` 已记录"bwd 从未被逐元素数值判过"
- 其 B7 段：`t1 = A^T dVnew`（gemm transpose_A）；`dV = beta ⊙ t1`；卡上 dV absmax 1.748 vs golden 0.485（3.6×）
- 下一主攻：**scan2b 逐箱审计**（输入已全对，可单点定位）：① dump `t1`（A^T dVnew）与 host 复算比；② 核对 `be_ub`（beta）装载；③ 逐 B7–B11 段与 golden 对拍
- DH0 单独列项（alt 形式已达 1.06e-1，需定 B6b 形式之争）

## [58] scan2b 定位到"读到的 WS_DV 与 scan2a 写的不一致"

### 零卡时判定（用归档的 w11box_out2.npy = 卡上 WS_DV，h0=on）
| 候选 | rel vs golden dv |
|---|---|
| **beta_row(Aᵀ dv_card)** | **4.194e-04** ✅ |
| Aᵀ(beta_row dv_card) | 6.549e-02 |
| beta_col(Aᵀ dv_card) | 6.235e-01 |
| A dv_card | 1.300e+00 |
⇒ **公式正确（dv = beta ⊙ Aᵀ dVnew）且 card 上存的 WS_DV 正确**；scan2b 内部再算一遍却得 absmax 1.748（golden 0.485，3.6×）
⇒ **scan2b 实际读到的 WS_DV ≠ scan2a 写入值**（切片/时序/adapter 传递）

### 尝试与回退
- 把 `T.copy(WS_DV[...], bq_l1)`（GM→L1 直读）改为经 `io_bq2` 中转 ⇒ **nan 复发**（io_bq2 在同段被 V/beta 装载复用）
- 已回退到直读（finite、3.6×），提交 `259bf907`

### 下一步（G7 第一步）
在 scan2b 内**直接 dump 它读到的 dVnew**：改为 GM→UB→L1，其中 UB 用**专用 debug 槽**（不碰 io_bq2），dump 到 WS_DGAM2 的空闲行或新增条件；与 box 的 out2 逐元素对拍，即可判定是"读错"还是"GEMM 错"。
- 严禁再用共享中转（io_bq2/io_bq/io_cc 均被复用），这是 [49]/[57] 的同一教训

## [59] 修订（2026-10-21 下午）：把被推翻的断言改成真话

> 按断言面扫描：只改"已写下但与现事实不符"的条目，历史测量不改写只加条件。

| 条目 | 原断言 | 现事实 | 处置 |
|---|---|---|---|
| [50] | "dVnew 写回前 ≈0，病灶在 GEMM 输入侧" | **假象链的延续**：当时 dump 被 dKt 零写覆盖；真实是 P 的 Gamma 双重 exp（[56]） | 结论作废；[56] 已取代 |
| [51] | "dVnew 真值含 3456 个 nan，唯一共同源头" | 该读数在 **D16 未修**时取得。D16 修后 `out2_WS_DV=6.785e-04`、NaN 归零 | 保留原测量，**加条件**：D16 未修状态；不得作为当前结论引用 |
| [49] | "三次 de-alias 皆负 ⇒ 病灶是跨核交接缺时序保证" | 方向正确，但**正向解已找到**：L0C→UB 直拷（`dq1_l0` 先例），不需要拆缓冲 | 补一行"正向解"；[46]/[56] 为凭据 |
| [48] | "dW≈0 ⇒ dVnew≈0" | 当时 dump 被覆盖；dW 现为 4.756e-04 | 已在 [51] 更正，此处并入 |
| [56] 表中 "DUMP-P nan=1397" | — | 成立，但属 **D16 未修**时的测量 | 不加删，作为 D16 的根因证据保留 |
| [57] | "全链 nan 已消灭" | ✅ 成立（nan → 有限错值 6.73/1.60/15.3/2.74） | 保留 |
| [58] | "scan2b 读到的 WS_DV ≠ scan2a 写" | ✅ 成立（host 复算 4.2e-4 vs 卡上 3.6×） | 保留，G7 起点 |
| design §18 | C1–C5 | 成立；本轮新增 **C6**（fp32 UB→GM 直写被拒）、**C7**（多次写的 GM 中转首读竞争）、**C8**（共享 UB 中转带 nan） | 已写入 §18.1，并在 §18.3 增两条"已否" |

### 本轮新增的两条纪律（进 §18.3）
1. **替换算子必须同 commit 删除被替代的表达式**——plan-10c 换了 GM 读却留下 `T.exp`，就是 D16 的来源。
2. **中转 lane 必须专用**——`io_bq/io_bq2/io_cc/t1_ub` 在混合核内都是多写者，拿它们做中转等于把别人的 nan/0 搬进来。

### 编号归属（防混淆，已写入 design §18.4）
D1–D4/D7 = 平台缺陷（对外 dossier）；D8/D13/D15/D16 = 本算子缺陷；D9–D12/D14 = 已证伪的中间假设。

## [60] 下次上机作战卡（2026-10-21 备妥）

### 已就绪的两处修复（本地，未上卡）
| # | 修复 | 依据 | 期望 |
|---|---|---|---|
| F1 | **B6b 用 dVnew 代替 Vnew**（`vnew_l1`→`bq_l1`） | CPU golden 判定：`e_do^T Q − dvnew^T W` **rel=1.515e-05**（zero/on 两 case 同值）；`vnew` 形式 4.3e-1 | `out1_DH0` 4.5e-1 → ~1e-4 |
| F2 | **t1 的 B 操作数专用 L1**（`dv_b_l1`，`bq_l1` 只作 A） | `bq_l1` 注释即 "A operand"；同一缓冲既作 A（dA）又作 B（t1）；A/B 的 L1 布局不同 ⇒ 卡上 dv 有限但 3.6×（[58]） | `ONE[basic] dv` 1.60 → <5e-3 |

### 车序（一条命令一车，禁止并发）
```bash
# 车 1（约 5 分钟）：两个修复的判决
bash scripts/w11_car.sh "boxb one"
# 车 2（约 8 分钟，仅在车 1 的 dv 达标后）：多 chunk + B·H=2
bash scripts/w11_car.sh "onet nd"
```
前置：`w11_car.sh` 已内置三档输入门（fail-closed）、清单 md5 拒跑、`flock` 单锁、自动归档。

### 判读表
| 观察 | 结论 | 动作 |
|---|---|---|
| `out1_DH0 rel < 5e-3` | F1 成立 | 勾 G8 |
| `out1_DH0` 仍 ~4.5e-1 | F1 未生效（编译缓存/写法） | 查生成码是否真的用了 bq_l1 |
| `ONE[basic] dv rel < 5e-3` | F2 成立（A/B 布局冲突确认） | 勾 G7 第一步，转 dbeta/dg |
| `dv` 仍 ~1.6 或 nan | F2 不是根因 | 走 G7 探针（专用槽 dump dVnew），**禁共享 lane** |
| `dk/dbeta/dg` 仍大 | 取决于 dKt(未实现)/dA 链 | 先看 `nd` 的 `s2b_dk/dbeta/dg` absmax 是否落回 golden 量级 |
| 任一腿 `RuntimeError: Compilation Failed` | 新 alloc 触发预算/布局问题 | 回退 F2，保留 F1 |

### 风险与回退
- F2 新增 L1 8 KiB（L1 总 512 KiB，当前 ~73 KiB，安全）；但**任何新增 alloc 都可能改 liveness**（C4），故 F1/F2 分属不同 kernel，可分别判定。
- 若 F2 引起编译失败或新 nan：`git revert` 该提交即可回到"dv 有限但 3.6×"的可诊断态。
- 车 1 只跑 boxb+one（不上 fp32、不上 typical），避免混淆。

### 上机前自检（已跑，零卡时）
- `python3 _process/_w11_gate.py` → `GATE-PASS`
- `python3 -m py_compile gdn_bwd_v2.py` → OK
- 两处改动均只动 scan2a/scan2b 的局部路径，`design.md §18` 已记录 C6–C8

## [61] 10-21 下午三车：F1/F2 双败、seed 未固定、多 chunk 是主病灶

### h1 车（boxb one onet zero nd，新实例）
- **F1（B6b 用 dVnew）无效**：DH0 4.581e-01（未改前 4.536e-01）
- **F2（dv_b_l1 专用 lane）无效**：`dv = 4.847e-01` = golden absmax ⇒ **t1 全 0**
- **重大流程发现**：`_case_inputs` **不固定 seed**，每次数据不同（alt 形式 1.060e-1 → 7.830e-2）⇒ **跨车数字不可比**；已给 4 个探针驱动加 `torch.manual_seed(7)`
- **流程错误**：车运行期间修改了正在执行的 `w11_car.sh`（bash 按字节偏移读脚本）⇒ `syntax error: unexpected end of file`，[4] pull 未执行；已改为"发车后不改正在跑的脚本"
- **教训修正**：F1 的"无效"结论**不可靠**（无 seed，前后数据不同），需在固定 seed 下重测；F2 的"t1 全 0"同车可比，**可靠**

### h2 车（回退 F1/F2 + 固定 seed，`boxb one onem`）
**boxb 基线**（basic fp16）：DQ 4.918e-04、**DH0 4.581e-01（alt 1.060e-01）**、DV 6.785e-04、DK 4.721e-04、DGAM2 5.212e-04、DKT/DE/DGAM/DW=0
**one（basic）**：
```
dq 1.940e-03 ✓   dk 3.701   dv 4.847e-01
dbeta 4.118      dg 4.014e-01   dh0 8.989e-02
```
- **勘误（16:05）**：`rel` 是 `maxdiff / gold_max`，不是 absmax。`dv rel=0.4847` ⇒ maxdiff=0.235（gold 0.4847），**不是全 0**；dk/dbeta 的 maxdiff 分别 ≈13.7 / 17.0（约 4× gold）
- ⇒ scan2b 的输出是**有限但错**（与 10-18 的 6.73/1.60/15.3/2.74 同类），此前"全 0"的判断**作废**
- `nd` 的 `dkg=7.03 / dvg=2.29 / dbeta=28.0` 是同量级的非零值，**两者一致**，不存在"nd 有值 / wrapper 全 0"的矛盾
- **推论修正**：F2 前后 `dv rel` 都是 0.4847 ⇒ F2 **无效但不有害**；F1 前后 DH0 都是 4.581e-01 ⇒ F1 **确实无效**
- H1（同步假设）随后被 h3 直接否证：加 `torch.npu.synchronize()` 后 one 结果**逐位不变**

**onem（1×1×256，4 chunk）**：
```
dq 2.194e-01   dk/dv/dbeta/dg = nan   dh0 6.786e-02
```
- 对照 basic（单 chunk）dq 1.940e-03 ⇒ **多 chunk 本身即病灶**（与 H=2 无关；typical 1.670e-01 同量级）
- 已给 `nd` 驱动加 `l0_multi_fp16` case 与车腿 `ndm`，一车拿逐箱首现

### h3 车（已发：`one onem ndm`）
判读：H1（sync）→ dk/dv/dbeta 是否从 0 变非 0；ndm → multi 的第一个 nan 箱（ws_g/ws_a/ws_h/s2a_out0..8/s2b_*）

## [62] ndm（multi 1×1×256）逐箱：scan2a 九箱全部干净，偏差在 scan2b / wrapper 链

```
CASE l0_multi_fp16
ws_g   nan=0 inf=0 absmax=3.408e+01
ws_a   nan=0 inf=0 absmax=1.000e+00
ws_h   nan=0 inf=0 absmax=6.255e-01
s2a_out0..8  全部 nan=0 inf=0
  out0 5.945e+00  out1 5.415e-01  out2 1.421e+00  out3 8.740e-01
  out4 0（dKt 未实现）out5 3.051e+00 out6 5.966e+00 out7 2.632e-13 out8 6.319e+00
s2b_dkg nan=0 inf=0 absmax=7.750e+00
```
- **多 chunk 下 scan2a 的九个箱没有 nan/inf**，量级与单 chunk 同阶
- 但 `onem`（走 wrapper）的 dk/dv/dbeta/dg 是 **nan** ⇒ 差异在 **wrapper 的 kernel 链**
- nd 的差异：**每个 kernel 之间都有 `torch.npu.synchronize()`**（cumsum/solve_a/scan1/scan2a 后共 4 处）
- h3 只加了第 4 处（scan2a→scan2b），basic 结果逐位不变（H1 在 basic 上否证）
- **h4 车（已发）**：wrapper 补齐 4 处 sync，跑 `one onem`；判读：multi 的 dk/dv/dbeta/dg 是否由 nan 转有限、dq 是否改善

### 若 h4 成立（预期）
根因 = **kernel 之间的 GM 依赖未被 stream 建模**（平台/adapter 层），而非 kernel 内部。长期处置：保留显式 sync（5 kernel 开销可接受），或查 adapter 是否为每次 launch 换 stream 后回写 design §18。

## [63] h4/h5：sync 消除 multi 的 nan；dq 的误差是"逐 chunk 递减"的确定性误差

### h4（4 处 `torch.npu.synchronize()`）
- multi：dk/dv/dbeta/dg 由 **nan → 有限**（8.601 / 2.064 / 18.76 / 4.572）⇒ **竞态确认存在**，sync 有效
- basic：dv 由 0.4847 → **0.6908**（数值变了但没修好）⇒ `synchronize()` 改变了执行时序
- dq 2.194e-01 **逐位不变** ⇒ dq 的误差不是竞态

### h5（4 处 sync + item() D2H；one 驱动加逐 chunk dq）
```
DQCHUNK[multi] t00 maxdiff=1.761e-03 goldmax=3.915   ← 精确
DQCHUNK[multi] t01 maxdiff=2.194e-01 goldmax=5.008
DQCHUNK[multi] t02 maxdiff=5.759e-02 goldmax=5.570
DQCHUNK[multi] t03 maxdiff=1.056e-01 goldmax=5.946
```
- **chunk 0 精确**（其 H_in 就是 h0 本身），chunk 1–3 相对误差 1–4%
- ⇒ **H2 假设：scan1 的跨 chunk 状态递推（WS_H）逐 chunk 引入误差**

### h6（已发：`ndm`，新增 S1G 逐 chunk 校验）
在 `_w11_num_diag.py` 里用 `_np_chunk_fwd` 的 per-chunk `hin` 与卡上 `WS_H[...,cc]` 逐 chunk 比：
- 若 `S1G cc=0` 精确、`cc=1..3` 递增误差 ⇒ H2 成立，且是 **fp16 状态递推的精度损失**
- 若全部精确 ⇒ H2 否决，误差在 scan2a 的逐 chunk 逻辑（下一步查 dqkt 的 Gam/P 逐 chunk）

## [64] h6–h8：两个 chunk 输入假设均被否决（WS_H 5e-4、GAM 1e-8）

```
GAM cc=0 rel=6.788e-09   cc=1 2.002e-08   cc=2 1.804e-08   cc=3 6.479e-08
S1G cc=0 maxdiff=0       cc=1 6.463e-04   cc=2 7.287e-04   cc=3 5.102e-04
```
- `WS_GAM`（cumsum 核，plan-10c）逐 chunk **精确到 1e-8** ⇒ H3 否决
- `WS_H`（scan1 递推）逐 chunk **5e-4~7e-4**（fp16 固有精度）⇒ H2 否决
- 但 `one` 的 dq 在 chunk 1–3 仍错 **1.0% / 4.4% / 1.8%**（t00 精确 1.8e-3）

### 由此确定的边界
dq = `dqkt @ K`（B4）+ `e_do @ H_in`（B5b），其两个 chunk 依赖输入（Gam、H_in）都已证准 ⇒ 误差只能来自：
1. **dq 两条分项本身的逐 chunk 计算**（`dq0_l0/dq1_l0` 的 init/累加，或 B4/B5b 的 hh 分块）
2. **dq 的写回地址**（chunk 间行区间重叠——但 `DQ[b,h,cc*C,...]` 逐 chunk 独立）
3. 反向扫描下 `cc == chunk_num-1` 的特殊分支影响了其它 chunk（唯一与 chunk 0 不同的路径）

### 下一步（下窗第一件事）
把 `boxb` 扩展为 **per-chunk box**（`_np_chunk_fwd` 的 cache 是 per-chunk，可直接给 golden）：
- 一次给出 multi 下**每个 chunk × 九个箱**的 rel，t01–t03 的误差会立刻落到具体箱
- 若落不到九箱（九箱全对而 dq 错），则误差在 dq 的两项累加（B4/B5b），再拆两项

## [65] h10 验收：DQSPLIT 把 dq 拆成 B4/B5b 后，误差仍在"两项组合"

```
DQSPLIT cc=0 card_vs_gold=4.500e-04  card_vs_B4=4.500e-04  B4max=3.915  B5max=0.000
DQSPLIT cc=1 card_vs_gold=4.381e-02  card_vs_B4=3.695e-01  B4max=5.008  B5max=1.726
DQSPLIT cc=2 card_vs_gold=1.034e-02  card_vs_B4=1.687e-01  B4max=5.571  B5max=0.947
DQSPLIT cc=3 card_vs_gold=1.777e-02  card_vs_B4=3.264e-01  B4max=5.946  B5max=1.958
```
- host 复算的 `B4+B5b` 与 test 的 golden 同源，卡上 `card_vs_gold` 与 one 驱动的逐 chunk 完全一致（4.4%/1.0%/1.8%）✓
- **cc=0：`B5max=0`（h0=zero ⇒ H_in=0），而 `card_vs_gold=4.5e-4`** ⇒ 纯 B4 项在 cc=0 精确
- cc=1–3：`card_vs_B4` 17–37%、`card_vs_B5 ≈ 1.0` ⇒ 卡上确实包含两项，误差在组合里

### H5（下一轮的检验，代码已写）
若 scan2a 读到了**相邻 chunk 的 H_in**，用 `hin[cc±1]` 重算 B5b 会把卡上 dq 精确复现：
- `card_vs_prevH` / `card_vs_nextH` 已加入 `DQSPLIT` 行；任一 ≈1e-3 即命中
- chunk 0 的 B5b 恒 0 ⇒ 位移在 cc=0 不可见，与"只有 0 精确"的观察自洽

## [66] D20：Vnew 的两条 `lcb` 交接在同一迭代内竞争（单 chunk 永不可见）

### 静态取证（`u0_l0` 的完整生命周期）
| 行 | 动作 | 说明 |
|---|---|---|
| 764 | `gemm(A, βV) → u0_l0 (init=True)` | U0 |
| 765 | `u0_l0 → workspace_lcb` | 第一次 W |
| 766 | `workspace_lcb → wh_ub` | 第一次 R（wh = U0） |
| 820 | `gemm(w_full, H_in^T) → u0_l0 (init=(hh==0))` | Wterm |
| 821 | `u0_l0 → workspace_lcb` | **第二次 W（覆盖 U0）** |
| 822 | `workspace_lcb → t1_ub` | **第二次 R** |
| 824 | `wh_ub += t1_ub` | Vnew |

- cv-sync 为一条 lane 建**一对** flag；同一迭代内的第二次 W-R **没有第二对** ⇒ 第二次 R 与 AIC 的 W 竞争（C7 的同型）
- **单 chunk 下 `Wterm ≡ 0`**（H_in = h0 = 0）⇒ 该路径从未被验证，basic 全绿也解释得通
- dq 经 `dP = dO·Vnew^T` 依赖 Vnew ⇒ 正好对应 chunk 1–3 的 1–4% 误差（chunk 0 的 Wterm 为 0）

### 修复（`3d0d64ad`）
两处交接改 **L0C → UB 直拷**（`dq1_l0` 先例，已在 [56] 验证）：
`T.copy(u0_l0, wh_ub)` 与 `T.copy(u0_l0, t1_ub)`，不再经过 `workspace_lcb`。

### 待验（h14）
`boxb one onem`：
- `DQCHUNK[multi]` 的 t01–t03 应从 1–4% 落到 <5e-3
- `out1_DH0`（B6b 也用 w_full/hin）应同步改善
- 若 `DQCHUNK` 仍差 ⇒ D20 不是全部原因，下一步用 h13 的 Vnew 探针（`ndv`）量化

## [67] D20 修复被编译器否决（D21）；本轮收口

### h14（D20 修复版）编译失败
```
TVMError: Mismatch in sync points between cube and vec for workspace workspace_2:
cube has 1, vec has 0
```
- 把 `u0_l0 → workspace_lcb → UB` 改成 `u0_l0 → UB` 直拷后，cv-sync 的 cube/vec 配对失衡
- **D20 的诊断成立**（同一 GM lane 一轮内两次 W-R，只建一对 flag），**但其修法需要 pass 侧支持**
- 已回退（`030ab072`），kernel 回到可编译态

### 本轮的净结论（10-21 下午，h1–h14 共 14 车）
| 发现 | 证据 | 状态 |
|---|---|---|
| wrapper 的 kernel 链存在竞态 | 4 处 sync 使 multi 的 nan → 有限 | 已修（sync + D2H） |
| **数据未固定 seed** | alt 形式跨车漂移 | 已修（4 个驱动 manual_seed(7)） |
| multi-chunk 的 dq 误差是确定性的 | 加 sync 后 2.194e-01 逐位不变 | 待修 |
| dq 的 B4 项在 chunk 0 精确 | `B5max=0` 时 `card_vs_gold=4.5e-4` | — |
| WS_H 递推、GAM 逐 chunk 都准 | 7e-4 / 1e-8 | H2/H3 否决 |
| 相邻 chunk 的 H_in 未发生错读 | prevH/nextH 均 rel=1.0 | H5 否决 |
| **D20**：Vnew 的第二条 `lcb` 交接无 flag | 单 chunk 下 Wterm≡0，恰好只在多 chunk 暴露 | 需 pass 支持（D21） |
| D17：A/Aᵀ 同 lane | DH0 停在 4.58e-1 | 需专用 lane |

### 下一步（下窗）
1. **D20 的可编译等价修法**：给 Wterm 换一条**已有且同形**的 GM lane（`workspace_vcb`/`workspace_vdk`，DV=DK 时同形），避免同一 lane 两次 W-R，同时保住 cv-sync 的配对（不再走直拷）
2. D17 的专用 Aᵀ lane（同 F2 的"新 lane 不接线"风险，需先验证一个最小 GEMM）
3. scan2b 的 dv（basic 仍 0.4847）：等 multi 修好后再回到 G7

## [68] lane 审计：`lcb` 每迭代承载 4 组 W-R，cv-sync 只建 1 对旗标

| lane | W | R | 说明 |
|---|---|---|---|
| **workspace_lcb** | **4** | **4** | U0 / Wterm / QH / dVnew 四组共用 |
| workspace_vcb | 7 | 7 | βV、Vnew、dO 等 |
| workspace_ldk | 5 | 5 | dW 中转 + W stash |
| workspace_lcc / vcc | 2–3 | 2–3 | raw/dP、P/dA 中转 |
| dq0 / dq1 / lbd / lc16 / vdk / w | 1–2 | 1–2 | — |

- **cv-sync 按 buffer 建一对 flag，不按访问次数** ⇒ `lcb` 的 4 组里只有第一组有序，后三组都可能竞争
- **"换 lane"无效**：所有候选 lane 在 scan2a 内都已被用过，换过去就变成同样的多组共用
- **可行模式 = 按组剥离**：每次把一组改成 **L0C→UB 直拷**，且**保留至少一组**在 lane 上（配对仍成立，不触发 D21）

### 渐进剥离计划
| 车 | 改动 | 期望 |
|---|---|---|
| h15 | 去掉 U0（剩 Wterm / QH） | 编译通过；dq chunk 1–3 明显改善 |
| h16 | 再去掉 Wterm（剩 QH） | Vnew 完全正确，dq multi 到 fp16 噪声 |
| h17 | QH 视编译情况 | QH 只影响 `de`（basic 下 golden=0） |

## [69] h15→h16 判决：剥离行不通，合并可行但 **dq 误差与交接无关**

| 车 | 改动 | 结果 |
|---|---|---|
| h15 | 只剥离 U0（L0C→UB 直拷） | **编译器卡死**（cache 7 分钟不增长，人工 kill） |
| h16 | **合并**：Wterm 累加进 U0 的同一 L0C（`init=False`），lcb 只用一次 | 编译通过；boxb 九箱与基线**逐位相同**；**multi 的 dq 逐位不变（2.194e-01）** |

- ⇒ **D20 的交接竞争不是 dq 误差的原因**（合并后逐位不变）
- ⇒ 1–4% 的误差在 **Vnew 的计算本身**（`W·H_inᵀ` 项；cc=0 时该项恒 0，故不可见）
- h16 的合并本身仍是**正向改进**（lcb 少一组 W-R、少一次 GM 往返），保留
- h17（已发）：`ndv` 用探针量化 **cc=1 的 Vnew** 与 fp64 的偏差
  - 若 Vnew 错 ⇒ 查 `w_full_l1[hh]` 的 3-D 子视图是否被 GEMM 正确读取（F2 的"子视图/新 lane 不接线"同型），修法 = python `range` 展开为独立 2-D L1
  - 若 Vnew 对 ⇒ 误差在 B4 的 dqkt（`P`/`dP`）路径

## [70] ★ h18 判决：Vnew 错 4.19%，与 dq 的 4.38% 同源 —— 因果链闭合

```
VN  cc=1 maxdiff=1.057e-01 goldmax=2.521e+00 rel=4.193e-02
DQCHUNK t01 maxdiff=2.194e-01 goldmax=5.008e+00  (=4.38%)
```
- Vnew 的相对误差与 dq 在 chunk 1 的相对误差**几乎相同**（4.19% vs 4.38%）
- cc=0 时 `W·H_inᵀ ≡ 0`（H_in=h0=0）⇒ Vnew 精确 ⇒ dq 精确（4.5e-4）；cc≥1 时该项非 0 且错 ⇒ 全链错
- **结论：Vnew 的 `W·H_inᵀ` 项是 dq 多 chunk 误差的根**

### 探针准入纪律（本轮第二次踩同一坑）
`WS_DKT` 有 **4 个写者**（dKt 零写、dqkt dump、P dump、Vnew dump）。第一次 VN 探针读到全 0，是因为 dKt 零写的放行条件没排除 `_S2A_VN_DBG`。**规则：任何新探针上线前，必须 grep 目标槽的全部写点，并把新开关加进每一个的放行条件**（`e59ffabd` 修复）。

### 下一步（下一轮第一件事）
在 `nd` 的 VN 段扩展一项**零卡时判定**：打印 `(Vnew_card − U0_gold)` 与 host 的 `W @ H_inᵀ` 的关系：
- 若 `Vnew_card − U0_gold` 与 `W@Hᵀ` 差大 ⇒ **该 GEMM 项错**，查 `w_full_l1[hh]` 的 3-D 子视图（bk_num=1 时等价 2-D，故更可能是 `transpose_B` 与该 A 布局不匹配）
- 若两者一致而 `U0` 错 ⇒ 转向 U0（`A @ βV`）

## [71] ★ VNSPLIT：卡上的 `W·H_inᵀ` 几乎没被加进去 ⇒ A 操作数 lane 问题

```
VNSPLIT cc=1  card_vs_u0=4.163e-03  wt_card_vs_host=1.032e+00
              vnew_vs_gold=4.193e-02  u0max=2.521  wtmax=1.024e-01
```
- `card_vs_u0 = 4.2e-3`：卡上 Vnew **几乎等于 U0**；`W·H_inᵀ` 应有 1.0e-1 量级，实测贡献 ≈ 0（相对误差 103%）
- **B5b 用的是同一个 `hin_l1` 且正确** ⇒ 装载无误 ⇒ 问题在 **A 操作数**
- 机理：`w_full_l1` 是 **3-D box**（`[bk_num, C, BK]`），且 **B6b 把它当 B 操作数**；一条 L1 lane 无法同时满足 A 与 B 的布局 ⇒ Vnew 的 GEMM 静默贡献 0（D22，D17 的同源）
- **修复（h20，`739c9934`）**：新增 `w_a_l1`（每 half 一条 **2-D** L1，python list + `range` 展开），同一 GM 源装载，只服务 Vnew 的 GEMM；`w_full_l1` 保留给 B6b 的 B 用法
- **h20 车（已发）**：`boxb onem ndv`
  - `boxb`：单 chunk 九箱不得回退
  - `onem`：`DQCHUNK` 的 t01–t03 应从 1–4% 落到 <1%
  - `ndv`：`wt_card_vs_host` 应从 1.03 落到 <1e-2

### 若 h20 成立，D17（DH0）很可能同源
B6b 的 `dVnew^T (-W)` 用 `w_full_l1` 作 **B**，而它同时也是 A 的候选 —— 若 h20 后 DH0 仍 4.5e-1，说明 B 侧也需一条专用 2-D lane（`w_b_l1`），与 A 侧对称。

## [72] ★★ 社区调研命中：多 chunk 全部误差 = 三处切片写法（官方 issue #1198）

### 社区实证
| 现象 | 官方出处 | 结论 |
|---|---|---|
| `Mismatch in sync points between cube and vec` | **#1019**（closed）+ 本地 `ascend_combinecv.cc:380` | 官方已知：workspace 由 `empty` 初始化、kernel 内再初始化 ⇒ cube/vec 操作数数量不一致。**修复未进入本 fork 版本**（断言仍在、无 `init="zeros"`） |
| `UB=>L1` / `L0C=>UB` 精度不过 | **#1198**（open） | 官方明确该场景在 Ascend C 后端有精度问题，workspace 消除 PASS 未覆盖 |
| `copy_gm_to_l1` 末参为 1（只搬 1 列） | **#1198 评论区** | `T.copy(ws[cid, :, k*BK], L1)` 的 extent 被解析成 `[1, M, 1]`；**正确写法 `ws[cid, 0, k*BK]` 或显式 range** |
| A/B 不能共用 L1 | **Ascend C Matmul 文档** | A1/B1 是独立缓冲、各按 NZ 分形对齐 ⇒ D22/D17 的修法有权威依据 |

### 本算子的命中面（38 处 GM→L1 装载中，仅 3 处是"`:` + 标量偏移"）
| 行 | 装载 | 被污染的项 | 单 chunk 为何不可见 |
|---|---|---|---|
| 698 | `workspace_w → w_full_l1` | Vnew 的 `W·H_inᵀ` | `H_in = h0 = 0` |
| 699 | `workspace_w → w_a_l1` | 同上（h20 的新 lane 也中招） | 同上 |
| 805 | `workspace_vdk → kt_l1` | B1 的 `Ktᵀ Δ'` | 末 chunk 的 `Δ'` 恒 0 |

⇒ **两项合计恰好多 chunk 的 dq 与 dvnew 链全部误差**；已按官方写法修正（`0e373514`）。

### 下一步（上卡即验，一车）
`boxb onem ndv`：
- `VNSPLIT wt_card_vs_host` 应从 1.03 掉到 <1e-2
- `DQCHUNK` t01–t03 应从 1–4% 掉到 <1e-2
- `boxb` 九箱不得回退

### 另一条待决
#1019 的修复（workspace `init="zeros"`）不在本 fork ⇒ 需要"kernel 内零化"的项（WS_D / WS_DKT / dGAM）继续手工置零；若要根治，需把 fork 升到含该修复的版本（单独任务）。

## [73] ★★ 社区深度调研：官方已有同族 bwd 蓝本，形态与我们差一个数量级

### 四条硬结论（均有原文/源码）
| 来源 | 结论 | 对我们的意义 |
|---|---|---|
| **#1019**（closed） | `Mismatch in sync points` 的成因 = workspace 由 `empty` 初始化 + kernel 内再初始化；官方已修，**本 fork 未包含**（`ascend_combinecv.cc:380` 断言仍在） | D21 只能继续用"每组至少一组留在 lane 上"的 workaround，或升级 fork |
| **#1198**（open） | `UB=>L1`、`L0C=>UB` 精度不过；`ws[cid, :, k*BK]` 的 extent 被解析成 `[1,M,1]`，须写 `ws[cid, 0, k*BK]` 或显式 range | 已修 3 处（`0e373514`）：W 与 Kt 的装载 |
| **#1799**（PR） | chunk_delta_h fwd：`use_g=False` 2-GEMM、**bf16 GEMM（"fp32 GEMM abandoned"）**、**M-form `M=exp(g_last)I - K^T W` 把 2-GEMM 折成 1-GEMM 累加**、**transit buffer 必须命名 `workspace*`**（`ascend_combinecv.cc:201` 的 `find("workspace")` 即此契约） | 印证 D7；印证我们 h16 的"合并 L0C"方向；命名契约我们已满足 |
| **#1802**（PR，今日仍在 review） | **chunk_o_bwd：单 `@tilelang.jit`、grid=256、chunks_per_block=32、0 GM workspace、无 T.Scope/flag/barrier、T.mma 显式 L0 staging、compensated GEMM（init=True+init=False 同 L0C）、L0/L1/L2/Boundary 全精度 PASS** | **可直接作蓝本**（源码已存 `_process/_ref/chunk_o_bwd_pr1802.py`） |

### 形态对照（官方已验证 vs 我们）
| 维度 | chunk_o_bwd (#1802) | gdn_bwd_v2（我们） |
|---|---|---|
| kernel 数 | **1** | 5 |
| GM workspace | **0** | 12+（workspace_idx）+ 10 个 `WS_*` 边界箱 |
| 同步 | **不手写任何 flag/barrier** | 手工 `barrier_all`/跨核交接 + 探针调试 |
| GM 切片 | **显式 range 19 处；`:`+标量 0 处** | 30 处 `:` 切片（3 处坏写法已修） |
| GEMM | `gemm_v0` 5 + **`T.mma` 显式 L0 staging 14** | 全 `gemm_v0`（隐式 L1 布局） |
| chunk 循环 | **块内 32 个 chunk 串行**（h/dh 作为输入） | 跨 block 全局扫描 + GM 递推态 |
| 代码量 | 583 行 | ~1500 行 |

### 结论（对"是不是很麻烦的节点"的回答）
- **不是无解，而是形态选错带来的复杂度**：我们把"h 递推 + 反向"拆成 5 核、12 workspace、10 个边界箱，每一处跨核/跨 kernel 交接都是一个潜在静默错误；官方的同族 bwd 全部塞进 **1 个 kernel、0 workspace**。
- **两条路**：
  - **短期（1 车）**：验证 h20 的切片修复 + 后续按官方纪律把剩余 `:` 切片改显式 range（机械、可枚举）
  - **中期（推荐）**：按 #1802 形态重构 scan2a/2b —— **h/dh 作为输入、块内 chunk 循环、0 GM workspace、T.mma 显式 staging**。问题面从"5 核 + 12 workspace"降到"1 核 + 0 workspace"，我们踩过的 8 类坑里至少 5 类直接消失。

## [74] 重规划（以终为始）：终态 → 五阶段 → 硬止损，见 openspec p2-13 §9

### 终态（先写死）
L0–L2–Boundary 数值门全过（fp16 主档）· 单文件 example · **≤2 kernel / ≤3 GM workspace / 零手写 flag**（对齐 #1802）· fp32 明确排除 · 真形档跑通并记录倍率 · 每项判决入库可复跑。

### 路径（倒推）
| 阶段 | 出口判据 | 车数上限 |
|---|---|---|
| P0 切片收口（`0e373514`） | onem 的 dq 逐 chunk <1e-2；ndv 的 wt_card_vs_host <1e-2；boxb 不回退 | 2 |
| P1 v2 收尾（scan2b dv、DH0/D17） | one/onem 六项 rel<5e-3 | 3 |
| P2 v3 骨架（#1802 形态，仅 P1 不达标时） | v3 的 dq 对拍 <5e-3 | 2 |
| P3 v3 全梯度 | 六项 <5e-3（basic+multi） | 3 |
| P4 交付化（分层测试 + 单文件 + 文档） | `--level all` PASS、ruff 干净 | 1 + 本地 |

### 止损（这是本轮最重要的一条）
- **P0+P1 超 5 车未收敛 ⇒ 立刻转 P2**，不再打补丁；
- **P2 超 2 车拿不到 dq ⇒ 降级"单 chunk only"交付**并把多 chunk 单列；
- fp32/bf16 不进本轮；每车走 `w11_car.sh`，车跑时**不改该脚本**。

### 与旧计划差异
旧 §8 是"逐项修 G1–G9"，没有止损；新 §9 以 DoD 倒推、每阶段设车数上限与降级出口，且把 v2 保留为 **oracle**（v3 与它并存对拍，不下线旧机制）。

### 下次上卡第一件事
`w11_car.sh "boxb onem ndv"` —— 一条命令同时给出 P0 的三个判据。

---

## [160] T3a 通过：真修成立、`BK=DK` 绕行已撤；快照版发车车脚本残缺已修（2026-10-25）

### 车 `20261025T0925Z`（腿 `wsc v3t ex`，840 s，全量日志已归档）

**T3a（`wsc`，exit 0）**

| 用例 | h_rel | a_rel | 判 |
|---|---|---|---|
| CONTROL H1/T256/DK64/BK64 | 3.621e-04 | 5.931e-05 | ok |
| CONTROL H1/T512/DK64/BK64 | 6.616e-04 | 5.398e-05 | ok |
| WSCMP H1/T256/DK128/**BK64** | 3.992e-04 | 5.961e-05 | ok |
| WSCMP H1/T256/DK128/BK128 | 3.992e-04 | 5.961e-05 | ok |

`BK=64` 与 `BK=128` **逐位相同**（两次都是 a 5.961e-05 / h 3.992e-04）⇒ `301bfde0` 的真修成立；
判据（a≈6e-5、h≈4e-4）同时满足；两条 64 形 CONTROL 也 ok ⇒ 量具自证。

**`v3t`（exit 1，红因是已知 Ratio 口径，非本轮引入）**

- T=64 六项 rel 全 PASS：dq 4.321e-04 / dk 3.669e-04 / dv 6.514e-04 / dbeta 3.400e-04 / dg 2.511e-04 / dh0 4.619e-04，与历史**逐位相同**（前置件改 idiom 不影响 64 形）。
- T=128 六项 rel PASS（dk 2.400e-03、dg 2.026e-03）；T=192 六项 rel PASS（dg 2.048e-03，历史 1.911e-03，差 7%，仍 <5e-3）。
- l2 T=65 正确拒绝；boundary 三条 WARN（非阻塞）；`LEVEL all ok=False` 来自 T=64 的 ratio 口径（[140] 已定性）。

**`ex`（exit 0）**：T=64 与 T=256 全 PASS，打印 `Kernel Output Match!`。

### 处置：撤 `BK=DK` 绕行

`gdn_bwd_v3.py` 与 `example_gdn_bwd.py` 各两处 `solve_a(B,H,T_len,DK,C,DK,...)` → `solve_a(..., 64, ...)`；
注释替换为 T3a 实测结论。`py_compile` 通过、`check_example_sync.py` 两段全绿（776/776 + SYNC2 3/3）。
撤绕行后的回归车见 [161]。

### 工具面新发现：快照版 `w11_car.sh` 残缺（已修，三处）

1. **连接参数**：把 key 拼成 `$REPO_ROOT/release/.venv/$CARD_KEY`、端口硬编码（`-p 31955`），且不再自加载 `.env.card` ⇒ 连着两次 `ABORT unreachable`（车把 ssh 的 stderr 吞了；手工同形 ssh 正常，才定位到拼接错误）。修为：自加载 `scripts/.env.card` + 五个变量 `:?` 校验 + `K="$CARD_KEY"` + `-p $CARD_PORT`。
2. **清单与腿表被裁**：MANIFEST 只剩 7 项、腿表缺 `wsc / rsa / v3t / ex` ⇒ 首车全部腿打印 `unknown leg`、20 s 空转结束。修为：MANIFEST 补 5 项（`_ws_cmp.py`、`_realshape_acc.py`、v3 三件），腿表补 4 条。
3. **判据行被过滤**：runner 与 pull 两处的 `grep -aE` 白名单不含 `RELAY / REALSHAPE-ACC / WSCMP / CONTROL / Kernel Output Match / PADINV / PRECISION_ / BOUNDARY_ / DKPART / LEVEL` ⇒ 腿跑完也拉不回判据。修为：两处白名单补齐，`run_leg` 摘要由 `tail -20` 放宽到 `tail -30`。

`bash -n` 通过后重发，12 项 md5 全 OK、车跑完 840 s。

### 归档

车 `20261025T0925Z`：**20 个文件 / 644 KB**（`w11car_all.log` + `w11car_{wsc,v3t,ex}.log` 全量 + 卡上 7 份运行器副本 + `MANIFEST.txt`）。
`w11_pull_logs.sh` 同样被快照裁掉，本次以手工归档脚本落盘（逐文件 `cat` + `md5 -q` + 字节数）。

---

## [161] 撤 `BK=DK` 绕行后的回归：数值零变化，`dk/dbeta` 与 BK 分块无关（车 `20261025T0952Z`）

腿 `v3t ex rsa`，1400 s，全量日志已归档（20 文件 / 652 KB）。

| 腿 | exit | 结果 |
|---|---|---|
| `v3t` | 1 | 红因仍是 T=64 的 ratio 口径（[140]）；l1 T=128/192 六项 rel PASS；l2 T=65 正确拒绝；boundary 三条 WARN |
| `ex` | 0 | `Test pass!` + `Kernel Output Match!`（T=64 / T=256） |
| `rsa` | 0 | 目标形 `ALL_PASS`；H1 扫描 T=128 `SOME_FAIL`、T=8192 `SOME_FAIL`（与历史同） |

**逐位相同**（对照 [154]/[156] 的历史值）：

- 目标形 H16/T8192：dq 5.897e-04 / dk 4.289e-03 / dv 1.043e-03 / dbeta 4.555e-03 / dg 4.718e-03 / dh0 4.642e-04 —— 六项一致；
- `v3t` T=64 与 T=256 六项、T=128 与 T=192 六项 —— 与首车（绕行版）一致；
- H1 扫描：T=128 dk 8.049e-03 FAIL；T=512 dk 1.321e-03；T=2048 dbeta 3.687e-03；T=8192 dbeta 6.116e-03 FAIL —— 全部一致。

### 两条推论

1. `BK=64` 与 `BK=DK` 在 `solve_a` 上等价（T3a 用 `wsc` 直接证过：A/H_in 两档逐位相同），故**撤绕行安全**，代码回到统一 idiom，不再背"回避不是修复"的注释。
2. **`dk`/`dbeta` 的薄边界不是 BK 分块造成的**——换回 `bk_num=2` 后两量在四个形状上与历史逐位同值。继续查的两个入口不变：`dk` 四项分解（T3'，腿 `dks128`）与 `dbeta` 主导项（T2b）。

### 计时口径（别误读）

目标形 `card_s=269.6`，首车同一用例是 44.9 ——本次 `TILELANG_CACHE_DIR` 在 launch 前被清空，**这个数含编译**，两者不可直接比较。纯运行时需热缓存复测。

### 工具面

归档脚本对 `w11car_ex.log` 抓到 0 字节（卡上为 2330 B、内容完整）；已重抓并给归档循环加非空校验（`[ -s ]` + 告警），避免"证据看起来归档了、实际是空文件"。
