# gap_F.md — p2-13 甲路 P1-3 波（执行代理 F）缺口登记

范围：GDN（Gated DeltaNet）**delta rule 前向递推 + 反向伴随递推**，910B(dav-2201) AIV 标量面。
容器 `cann910b-f`（trunk compat 快照 965 行 / md5 75a3b35b…，**未 cp trunk**，编排者正在改 §12）。
所有凭据可在 `attempts/F/verdict_log_F.txt` 逐行复核；复跑入口 `bash attempts/F/run_F.sh --matrix`。

---

## G-F0（**结论性，非缺口**）decay 的"log 缺件"在给定语义下不成立

tasks P1-3 原文写"decay 的 log 缺件=需查 exp 负指数表载或 §11 扩展件"。实测拆解：

- 任务书给的语义是 **g 以 log 域入参**：`a_t = exp(g_t)`，`g_t <= 0`。前向只吃 `exp`；
  反向求导链 `dg_t = da_t · a_t` 同样只吃 `exp`。两件 codegen 实测发射面：
  `math/intrin : expf`（verdict_log_F.txt:30/50/70/90/110/130/149）——**零新增 compat 依赖**，
  `expf` 是 §11(GAP-C) 已并入 trunk 真源的件（`port910b_compat.h:724`，宏 `#define expf(x)` :756）。
- 于是"exp 负指数表载"这条也不需要：§11 的 `tl910b_expf` 是**范围规约 + 6 阶 Taylor + 指数域拼 2^n**
  的纯算术实现，负指数天然覆盖（本波实测 a_t 单点 max_rel = **1.427e-07**，
  verdict_log_F.txt:247）。
- `logf` **确实缺**，但只在另一种入口口径 (b)「线性域给 a∈(0,1]、件内 g = log(a)」时才需要
  （fla 部分实现与 chunked 形态的 log 域累加和走这条）。差分已做成正/负形对照：

  | 形 | compat 状态 | 判决 |
  |---|---|---|
  | 负形 | 快照未注入 GAP-F（965 行） | `F-LOGF-PROBE-COMPILE-FAIL` + `tl_kernel.asc:10:9: error: use of undeclared identifier 'logf'` → `DIFF-NEGCONFIRMED`（:155-160） |
  | 正形 | 注入 GAP-F（965→1052 行） | `F-LOGF-PROBE-COMPILE-PASS`，`emitted_math=['expf','logf']`（:162-165） |

  缺口成立，交付件 `attempts/F/compat_patch_F.h`（87 行，幂等锚 `TL_PORT910B_COMPAT_GAP_F_H`）
  + 装配器 `patch_compat_F.py`（追加/还原/状态，只动容器 pip 副本，**未写 trunk**）。
  数值定标（numpy 逐行复刻 vs `np.log` fp64）：
  `DECAY-DOMAIN a∈(0,1] max_abs=4.812e-07 / max_rel(|ln|>1e-3)=1.844e-07`；
  `WIDE-SWEEP 2^k×[1,1.9] k∈[-30,30] n=610 max_abs=1.916e-06 / max_rel=1.140e-07`；
  关键点 `log(1)/log(e)` abs_err=0，`log(0.5)/log(2)/log(1e-30)` ≤ 5.6e-09。
  往返 `a→log→exp→a' max_rel=4.190e-07`（:168-178）。
- **P1-3 的 log 条目可如此销账**：口径 (a)（本波件采用）零缺件；口径 (b) 的 `logf` 已备补丁，
  等编排者波后合并即可，不阻塞 P2。

## G-F1（合并隐患，实测坑）GAP 块位视图 helper 同名重定义

trunk §11(GAP-C) 已占用 `tl910b_gap_f2u / tl910b_gap_u2f`（真源 :710-719 实测）。
任何新 GAP 块照抄这两个名字 → **重定义，整树编不过**。本波补丁的做法：

```cpp
#ifdef TL_PORT910B_COMPAT_GAP_C_H
#define TL910B_F2U(x) tl910b_gap_f2u(x)   // 复用 GAP-C
#define TL910B_U2F(b) tl910b_gap_u2f(b)
#else                                     // GAP-C 不在树内才自带 F 专名
__aicore__ inline unsigned int tl910b_gapF_f2u(float x) { … }
__aicore__ inline float tl910b_gapF_u2f(unsigned int bits) { … }
#endif
```

**给后续 GAP 块的纪律**：需要位视图就复用 GAP-C 那两个；要自带副本必须加域后缀（`gapF_`/`gapG_`…），
且用 `#ifdef <上游锚>` 判存在性，不要凭"应该是这样"。

## G-F2（结构性结论，非缺陷）前向可按值维切核，反向必须按 head 并行

delta rule 前向在值维 j 上**逐列独立**：`pred[j]`、`u[j]`、`o[j]` 只吃 state 的第 j 列 ⇒
工作单元 = `(head h, 值维列 j)`，共 `H*DV` 个，**零跨核归约**（`u // DV`、`u % DV` 反解，同 E 波 attn_sw 取模法）。
反向不同：`dq[i]=Σ_j S_t[j,i]·dO[j]`、`dk[i]=Σ_j(…)`、`dβ=Σ_j du[j](v[j]-p[j])`、`da=Σ_{j,i}` **全是对 j 的归约** ⇒
不可按列切，必须 **unit = head**、整块伴随态 `(DV,DK)` 常驻单核。件里已加静态校验
`if H % cores: raise`（bwd :69-70）。

这条直接决定生产形态的并行度上限：反向并行度 = head 数（P1 阶段 H=16 → 16 核）；
若 head 少于可用核数，需再切 token 段做分块 BPTT（本波未做，见"后续证真点"）。

## G-F3（**上卡必证**，唯一硬风险）pure 变体的 GM 就地 read-modify-write 读后写可见性

`pure` 变体（fwd/bwd 各一）零 `alloc_shared`，跨 token 状态/梯度就地存在 GM 工作缓冲
（fwd 用 `SOUT`，bwd 用 `DSOUT` + `DQ/DKG` 累加），codegen 实测发射
`tl::read_gm_bypass_dcache` + `tl::write_gm_bypass_dcache` 成对
（verdict_log_F.txt:49/89，特征行 `gm_rmw: True`）。

风险：delta rule 的递推**逐 token 依赖上一 token 写回的值**，若 bypass-dcache 的
读后写序列在 aicore 上不保证同地址可见性（无显式 fence/无队列序保证），数值会静默错。
本波**只有 CPU 同序复刻能证明算法对，证不了存储序**。

- 缓解：**首推 `ub` 变体**——UB 常驻，GM 只在首尾各碰一次，`gm_rmw: False`
  （:26/66/106/145），根本不吃这条风险；且全程 **零 `T.copy`**，也不吃 C 波 G-C4/G-C7
  未证真的 MTE 搬运单位/stride 面。
- 上卡证真点（单窗，≤2 形）：`ub` 先行拿数值绿；`pure` 仅在需要省 UB 时才验，
  验法是 T=1 与 T=2 两形对比（T=2 若 rel 突升即坐实读后写不可见）。

## G-F4（接口约束）反向依赖前向逐 token 落 HIST

`da_t = Σ S_{t−1}·dS'_t` 必须要 **S_{t−1}**。前件因此带 `emit_hist`，逐 token 写
`HIST (H,T,DV,DK)`；反件只读它。若宿主对前向传 `--hist 0`（省带宽），反向**不可用**。
⇒ 落 HIST 是 delta rule 可反传的**必要条件**，不是可选优化；P2 接入时 H0/O/SOUT/HIST 四出口的
分配量按 `(H,T,DV,DK)` 计入（T=128,DK=DV=64,H=16 时 HIST 单卡 fp32 = 33.5MB，需在此定 chunk 与否）。
（本波只证"依赖存在"，chunk 形态——即 tilelang 主仓 `examples/gdn` 的分块口径——留 P2。）

## G-F5（补丁已知边界）`tl910b_logf` 次正规数与 x<=0 的钳位取舍

- **次正规**（指数域全 0）：按 `e=-127` 处理，`m` 直接拼 `0x3f800000|mant` ⇒ 误差可达 O(1)。
  decay 值域 a∈(0,1] 不触及；若将来对任意 x 用，需另补次正规分支。
- **x <= 0**：返回 `-FLT_MAX` 而非 NaN/-inf。理由：910B 标量面无 NaN 传播契约，
  而 decay 语义下 `x<=0` 是非法输入，钳位常量比 NaN 更早在校验里暴露（实测
  `EDGE log(0)/log(-1) -> -3.40282e+38`，:175-176）。
  与 §11 `tl910b_expf` 的"下溢置 0 / 上溢置 FLT_MAX"是同族取舍，风格一致。
- `sqrtf/tanhf/powf/log2f/floorf` 仍缺（本波件**一个都不吃**）。

## G-F6（环境类，建议同步给 A/B/C/E 各波）容器推送会**静默留下旧版本**

`run_*.sh` 常用 `docker exec … bash -c 'cat > /tmp/x.py <<EOF'` 推送。本容器实测：
`docker cp` 落进 /tmp 的文件属主是 **uid=501(dialout)/644**，而 exec 侧虽 uid=0 root，
**DAC_OVERRIDE 被剥** ⇒ 对该文件**没有覆写权** ⇒ `cat >` 打
`bash: line 1: /tmp/f_gdn_delta_bwd_910b.py: Permission denied`，
而这一句在 `&&`/循环里**不会让 `set -e` 中断** ⇒ 随后编译的是**上一个版本**。

本波真实代价：第一次 `--matrix` 的 bwd 判决打在"伴随漏项修复前"的源码上（见 RESULT.md 偏差段），
靠 md5 比对才坐实：`host=4bab2619… cont=45d5eb30…`。

修法（已进 `run_F.sh:26-36`）：推送一律 `docker cp` + **逐文件 md5 PUSH-VERIFY**，不匹配 `exit 1`。
现日志 `PUSH-VERIFY SAME ×6`（:6-11、:218-223）。
**建议编排者把 md5 校验补进其它波的 run 脚本**——凡用 `cat >` 推送的判决都有同类风险。

## G-F7（口径记录）dtype 只取 fp32，未触 bf16/fp16

任务书允许"向量/标量面"，未要求精度面。C 波 G-C? 记录过 **bf16 标量 cast 后端缺件**（故取 fp16）。
本波 delta rule 递推对精度敏感（state 连乘衰减 + 反向跨 token 累加），**刻意只做 fp32**：
fp32 是上卡对拍的判据基线，半精度接入会叠加"量化漂移"与"递推漂移"两条误差源、失掉解释力。
⇒ 半/ bf16 版留生产接入波，届时需与 C 波的 bf16 标量 cast 缺口一并裁决。
