"""GDN（Gated DeltaRule）方言件：线性注意力的逐 token 递推前向 + 反向伴随递推（910B 标量面）。

【做什么】前向 `forward(q, k, v, g, beta, out_dtype, target)` 按头维护一份状态矩阵 S (dk, dv)，
逐 token 走五步：① 衰减 `S *= exp(g[t])`（g 是每键通道的对数衰减，这就是 "gated"）；② 用当前键
召回一遍 `pred = kᵀ S`；③ 算 delta 修正项 `u = beta * (v[t] - pred)`（"该写进去的" 减 "已经有的"）；
④ 秩一写回 `S += k ⊗ u`；⑤ 读出 `o[t] = q[t]ᵀ S`。交回 (heads, seq, dv) 的输出。
反向 `backward(q, k, v, g, beta, dout, target)` 走同一条递推的**伴随链**，交回 dq/dk/dv/dg/dbeta
（+ 可选的初状态梯度 dH0）。短卷积（conv 位点）在姊妹件 `gdn_conv_asc.py`，由 `gdn_kernel`
一并导出成 `conv_forward`——递推件只吃已经归一、已经卷过的 q/k/v 与算好的 g/beta。
【怎么做】① 这份状态是**跨 token 串行**的：递推关系不允许在时间轴上并行，所以两份正文都把 token
   循环写成串行；并行度只出现在**通道/单元轴**上。② CPU 那份用本地二维缓冲按格扫（P1-4 合流
   起一行未动，target=cpu 的语义对拍口径保持原样）；昇腾那份自 P1-4 起改成 **910B 的 AIV 标量
   面**：`T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial`，工作单元 = (头 h, 值维列 j)，
   状态一律按**转置**形态存/遍历（`Sd[h, j, i] ≡ S_std[i, j]`，i 是键维）——递推在值维 j 上逐列
   独立，按 j 切单元零跨核归约，转置后第 j 列是 dk 个连续元素，标量面顺址读写（凭据
   `attempts/F/f_gdn_delta_fwd_910b.py`）。③ 反向不能再按列切：dq/dk/dβ/dg 都是**对值维的归约**，
   所以反向的工作单元 = 头，整块伴随态 (dv, dk) 常驻单核，并强制 `heads % cores == 0`
   （结构性结论 G-F2）。④ 头数、键宽、值宽、核数、档位都是编译期常量，seq 是动态符号，
   网格上界只放常量（与 gemm/rope/add_ln/attn_sw 同一口径：缓存键不含 seq）。⑤ exp 走 §11
   软件 expf（trunk 已并），decay 门 g 以 log 域入参 ⇒ 只吃 exp、不需要 logf（F 波口径(a)）。
   ⑥ 全程 fp32。
【为什么】反向在 P1-4 之前标的是 `partial`（走 torch 自动微分），理由是"昇腾方言没有可对齐的
   写法"；这个前提已被 attempts/F 那一波自己写出来并证真：910B 标量面的六项梯度（dq/dk/dv/
   dβ/dg/dH0）全部实现，compile PASS + CPU golden（rel 5.76e-07，伴随式另经有限差分与 torch
   autograd 双独立锚）⇒ 本波把反向接进接口、撤下 partial（D-int2(a)）。被否方案（当初）：硬凑
   一份看着像 naive 的反向方言件——递推反向要再存一份跨 token 的伴随状态，形状与流水都要重设计，
   本地编不出就只能靠猜（红线 R14）；现在形态已实测可编，猜的部分消掉了，所以接。
   两处仍要如实留痕：① 数值只证到 CPU 同序对拍，卡上 rel 待与 P1-1d 同窗收（本波只判接口合流）；
   ② `pure` 档的状态就地读写下了"跨 token 可见性"的赌注（G-F3），CPU 证不了 ⇒ 首推 `ub` 档。
"""
import os
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 反向自 P1-4 起内核化：ascend target 走 910B 标量面伴随件（六梯 dq/dk/dv/dβ/dg/dH0 全实现）；
#: cpu target 仍走 torch 自动微分（那是语义对拍的尺子路，不是"没做完"）。档位与并行约束见
#: `ASC_VARIANTS` 与模块 docstring 的【怎么做】③。
BWD_STATUS = "kernelized"
#: `backward` 交回的梯度名（顺序与返回值一致；`return_dh0=True` 时末尾多一个 dH0）
BWD_GRADES = ("dq", "dk", "dv", "dg", "dbeta")
#: 昇腾侧可用档位：ub=状态/伴随态常驻 UB（首推）；pure=零 UB、就地读写 GM 工作缓冲（待卡证 G-F3）
ASC_VARIANTS = ("ub", "pure")
#: 环境开关：档位（默认 ub）、一次发射占用的向量核块数上限（默认 8，挑不整除就回落核数）、
#: 前向是否落逐步历史（默认落——反向要它）
ENV_VARIANT = "SYS1_GDN_VARIANT"
ENV_BLOCKS = "SYS1_GDN_BLOCKS"
ENV_EMIT_HIST = "SYS1_GDN_EMIT_HIST"
#: 缺省核块上限（910B AIV 核数远多于这个值，先按保守值起，P2 三栈联调时按实测调 ENV_BLOCKS）
ASC_BLOCKS = 8


def _asc_variant(variant: str | None = None) -> str:
    """挑 910B 档位：显式参数 > env > ub；认不得的写法直接报错，不许悄悄退回另一档。"""
    raw = (variant or os.environ.get(ENV_VARIANT) or "ub").strip().lower()
    if raw not in ASC_VARIANTS:
        raise ValueError(f"未知 gdn 档位 variant={raw!r}，只支持 {'/'.join(ASC_VARIANTS)}")
    return raw


def _pick_cores(units: int, limit: int) -> int:
    """在 limit 以内挑能整除 units 的最大核数（工作单元要均分给核，不整除就均分不了）。

    白话：活必须正好摊平给工人，摊不匀就少用几个工人，宁可少开几个核也不留半格没人管。
    """
    for c in range(min(units, limit), 1, -1):
        if units % c == 0:
            return c
    return 1


def _blocks_limit(blocks: int | None = None) -> int:
    """核块上限：显式参数 > env > 常量。"""
    raw = blocks if blocks is not None else (os.environ.get(ENV_BLOCKS) or ASC_BLOCKS)
    n = int(raw)
    if n < 1:
        raise ValueError(f"核块数必须 >=1，实得 {n}")
    return n


def gdn_cpu_impl(Q, K, V, G, Beta, O, heads: int, dk: int, dv: int):
    """CPU(c) 正文：每头一块，token 串行递推，通道轴扫本地缓冲。

    白话：每人守着一块小黑板（状态）。每来一条新记录：先把黑板整片按比例淡掉一点，再照当前键
    把黑板上已有的内容读出来当"预期"，用真值减预期得到"这次真正要补的"，乘上这一步的下笔力度
    写回黑板，最后照查询把黑板读一遍就是这一步的输出。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    seq = T.dynamic("seq")

    @T.prim_func
    def gdn_impl(Q: T.Tensor((heads, seq, dk), "float32"), K: T.Tensor((heads, seq, dk), "float32"),
                 V: T.Tensor((heads, seq, dv), "float32"), G: T.Tensor((heads, seq, dk), "float32"),
                 Beta: T.Tensor((heads, seq), "float32"), O: T.Tensor((heads, seq, dv), "float32")):
        """被追踪的那一层：头数、键宽、值宽是常量，时间轴走动态维并全程串行。

        白话：CPU 件这一份，每头守一块小黑板，每来一条新记录就走五步——整片按比例淡掉、照当前
        键把已有内容读出来当预期、真值减预期乘上下笔力度补进去、最后照查询读一遍就是这一步
        的输出。时间轴一步压不得，所以这条链只能串行走。
        
        """
        with T.Kernel(heads) as hh:
            S = T.alloc_local((dk, dv), "float32")
            kt = T.alloc_local((dk,), "float32")
            qt = T.alloc_local((dk,), "float32")
            vt = T.alloc_local((dv,), "float32")
            gt = T.alloc_local((dk,), "float32")
            dec = T.alloc_local((dk,), "float32")
            pred = T.alloc_var("float32")
            upd = T.alloc_var("float32")
            acc = T.alloc_var("float32")
            T.clear(S)  # 初始状态恒为零：本件不带跨段状态入口（跨段续推属上层 KV 的事）
            for t in T.serial(seq):
                T.copy(K[hh, t, 0], kt)
                T.copy(Q[hh, t, 0], qt)
                T.copy(V[hh, t, 0], vt)
                T.copy(G[hh, t, 0], gt)
                # ① 衰减：先把每键通道的衰减系数算出来，再整片淡掉黑板
                for kk in T.serial(dk):
                    dec[kk] = T.exp(gt[kk])
                for kk in T.serial(dk):
                    for vv in T.serial(dv):
                        S[kk, vv] = S[kk, vv] * dec[kk]
                bt = Beta[hh, t]
                for vv in T.serial(dv):
                    # ② 召回：照当前键把黑板读一遍
                    pred = 0.0
                    for kk in T.serial(dk):
                        pred = pred + S[kk, vv] * kt[kk]
                    # ③ delta 修正：真值减预期，乘下笔力度
                    upd = (vt[vv] - pred) * bt
                    # ④ 秩一写回
                    for kk in T.serial(dk):
                        S[kk, vv] = S[kk, vv] + kt[kk] * upd
                    # ⑤ 读出：照查询把写回之后的黑板读一遍
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + qt[kk] * S[kk, vv]
                    O[hh, t, vv] = acc

    return gdn_impl


def gdn_asc_impl(Q, K, V, G, Beta, H0, Hist, Sout, O, heads: int, dk: int, dv: int,
                 cores: int, variant: str = "ub", emit_hist: bool = True):
    """昇腾正文（910B / dav-2201 可编形态）：AIV 标量面，状态转置存，工作单元=(头, 值维列)。

    白话：一本账拆成"每个值维列一条链"，每条链一个人从头守到尾——小黑板（这一列的 dk 个数）
    摆在工人手边的台面上，每来一条记录先按各键通道的比例淡掉、顺手照键读出预期、算出要补的
    那笔、再补进去、最后照查询读出这一步的结果。列与列之间互不相干，所以列可以均分给工人；
    但时间轴一步压不得，所以每条链内部全是串行。

    四条 910B 方言纪律（P1-4 合流，凭据 attempts/F/f_gdn_delta_fwd_910b.py + RESULT.md）：
      * 载体只有 `T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial`；**不碰** `T.SimtVF`/
        `T.Parallel`/`T.Pipelined`/`T.copy`——该代际无 SIMT 硬件模型，这些件实测编不出
        （D-int1），零 `T.copy` 还顺带绕开了 MTE2/MTE3 的单位/stride 未证真面（G-C4/G-C7）。
      * 状态**转置存储**：`Sout[h, j, i] ≡ S_std[i, j]`（j=值维、i=键维），第 j 列的 dk 个数
        在内存里连着，标量面顺址；接口层不暴露这个朝向（H0/Hist/Sout 都是内部工作缓冲）。
      * 衰减系数按**每键通道**算（`exp(g[h,t,i])`，与本文件 CPU 正文与 `_eager` 同口径）：
        F 件那一份是每 token 一个标量门，合流到接口时按接口的 g 形状摊成通道向量，
        递推次序一个字母没改（先淡掉、同一趟里召回）。
      * shared 缓冲按档位分支声明且**都被访问**（G-C8：只写不读的死缓冲会落成命名空句柄）。

    :param variant: `ub`=这一列常驻 UB（首推，GM 只在进出各碰一次）；`pure`=零 UB、状态就地
        读写 GM 工作缓冲 `Sout`（省 UB 但赌"跨 token 的读后写可见性"，G-F3 待卡证）。
    :param emit_hist: True 时逐 token 落下进 t 之前的状态 `Hist[h,t,:,:] = S_{t-1}`——反向件
        求 dg（衰减的梯度）必须要这份历史，这是"前向落状态"的必要性凭据（G-F4，代价是
        (heads, seq, dv, dk) 这块带宽）。
    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，P1-4 起本地实编判 .o）。
    """
    import tilelang.ascend.language as T

    seq = T.dynamic("seq")
    units = heads * dv
    if units % cores:
        raise ValueError(f"工作单元 units=heads*dv={units} 必须能被 cores={cores} 整除")
    upc = units // cores

    @T.prim_func
    def gdn_impl(Q: T.Tensor((heads, seq, dk), "float32"), K: T.Tensor((heads, seq, dk), "float32"),
                 V: T.Tensor((heads, seq, dv), "float32"), G: T.Tensor((heads, seq, dk), "float32"),
                 Beta: T.Tensor((heads, seq), "float32"),
                 H0: T.Tensor((heads, dv, dk), "float32"),
                 Hist: T.Tensor((heads, seq, dv, dk), "float32"),
                 Sout: T.Tensor((heads, dv, dk), "float32"),
                 O: T.Tensor((heads, seq, dv), "float32")):
        """被追踪的那一层：头/键宽/值宽/核数/档位是常量，时间轴留动态维、链内全串行。

        白话：这一份是"每个值维列一条链"的摆法——列先摊给工人，每条链再从初值开始，
        一条一条记录按五步走完全程；步与步之间不能换顺序，所以最外层一定是串行。
        
        """
        with T.Kernel(cores) as core_id:
            a_i = T.alloc_var("float32", T.float32(0.0))
            bt = T.alloc_var("float32", T.float32(0.0))
            si = T.alloc_var("float32", T.float32(0.0))
            ki = T.alloc_var("float32", T.float32(0.0))
            qi = T.alloc_var("float32", T.float32(0.0))
            vj = T.alloc_var("float32", T.float32(0.0))
            pred = T.alloc_var("float32", T.float32(0.0))
            uu = T.alloc_var("float32", T.float32(0.0))
            oj = T.alloc_var("float32", T.float32(0.0))

            if variant == "ub":
                # 四个 shared 全被读写，不触发 G-C8 的命名空句柄
                st_ub = T.alloc_shared((dk,), "float32")   # 本单元（h, j）那一列的状态
                k_ub = T.alloc_shared((dk,), "float32")
                q_ub = T.alloc_shared((dk,), "float32")
                dec_ub = T.alloc_shared((dk,), "float32")  # 每键通道衰减系数 exp(g)

            with T.Vector():
                for rr in T.serial(upc):
                    u = core_id * upc + rr
                    h = u // dv
                    j = u % dv

                    if variant == "ub":
                        # 入口：初值 H0[h,j,:] → UB（标量循环，不用 T.copy ⇒ 零 MTE2）
                        for i in T.serial(dk):
                            st_ub[i] = H0[h, j, i]

                    for t in T.serial(seq):
                        bt = Beta[h, t]
                        vj = V[h, t, j]

                        if variant == "ub":
                            for i in T.serial(dk):
                                dec_ub[i] = T.exp(G[h, t, i])   # → codegen expf（compat §11）
                                k_ub[i] = K[h, t, i]
                                q_ub[i] = Q[h, t, i]

                        if emit_hist:
                            # 落下"进 t 之前的状态" = S_{t-1}（反向求 dg 要用）
                            for i in T.serial(dk):
                                if variant == "ub":
                                    Hist[h, t, j, i] = st_ub[i]
                                else:
                                    Hist[h, t, j, i] = Sout[h, j, i]

                        # 趟 1：① 衰减 + ② 召回（同一趟，pred 吃的就是已衰减的 S）
                        pred = T.float32(0.0)
                        for i in T.serial(dk):
                            if variant == "ub":
                                st_ub[i] = st_ub[i] * dec_ub[i]
                                pred = pred + st_ub[i] * k_ub[i]
                            else:
                                a_i = T.exp(G[h, t, i])
                                si = Sout[h, j, i] * a_i
                                Sout[h, j, i] = si
                                ki = K[h, t, i]
                                pred = pred + si * ki

                        # ③ delta 修正：真值减预期，乘这一步的下笔力度
                        uu = bt * (vj - pred)

                        # 趟 2：④ 秩一写回 + ⑤ 读出（o 吃的是已更新的 S）
                        oj = T.float32(0.0)
                        for i in T.serial(dk):
                            if variant == "ub":
                                st_ub[i] = st_ub[i] + k_ub[i] * uu
                                oj = oj + st_ub[i] * q_ub[i]
                            else:
                                ki = K[h, t, i]
                                qi = Q[h, t, i]
                                si = Sout[h, j, i] + ki * uu
                                Sout[h, j, i] = si
                                oj = oj + si * qi

                        O[h, t, j] = oj

                    if variant == "ub":
                        # 出口：末状态 UB → GM（标量循环，不用 T.copy ⇒ 零 MTE3）
                        for i in T.serial(dk):
                            Sout[h, j, i] = st_ub[i]

    return gdn_impl


def gdn_bwd_asc_impl(Q, K, V, G, Beta, DO, Hist, Lam, DQ, DKG, DV_, DBeta, DG, DH0,
                     heads: int, dk: int, dv: int, cores: int, variant: str = "ub",
                     dstate: str = "zero"):
    """昇腾反向正文（910B / dav-2201 可编形态）：伴随链沿 token 逆序，工作单元=头。

    白话：倒着查这本账。手上留一张"这份账后面还有多少没算的账"（伴随态 Λ），从最后一条记录
    往前推：先把这一步读出的那笔贡献并进来，再拆出"这一笔补写得对不对"（du）、"下笔力度该改
    多少"（dβ）、"每个键通道要改多少"（dk/dg），并把 Λ 推到上一步。因为 dq/dk/dβ/dg 都要沿
    **值维**把所有列合起来，反向没法按列分人——只能一个头一个人守到底。

    数学（逐 head，state 转置存 Sd[j,i] ≡ S_std[i,j]；Λ_t ≡ ∂L/∂S_t，t 逆序）：
        a_i = exp(g[h,t,i])；S'_t[j,i] = a_i·S_{t-1}[j,i]；p_t[j] = Σ_i S'_t[j,i]·k_t[i]
        u_t[j] = β_t(v_t[j] − p_t[j])；Λ^tot[j,i] = Λ_{t+1}[j,i] + dO_t[j]·q_t[i]
        du[j] = Σ_i Λ^tot[j,i]·k_t[i]；dp[j] = −β_t·du[j]；dv_t[j] = β_t·du[j]
        dβ_t = Σ_j du[j]·(v_t[j] − p_t[j])
        dq_t[i] = Σ_j (S'_t[j,i] + u_t[j]·k_t[i])·dO_t[j]
        dk_t[i] = Σ_j (u_t[j]·Λ^tot[j,i] + S'_t[j,i]·dp[j])
        dS'_t[j,i] = Λ^tot[j,i] + k_t[i]·dp[j]；Λ_{t−1}[j,i] = a_i·dS'_t[j,i]
        dg_t[i] = a_i · Σ_j S_{t-1}[j,i]·dS'_t[j,i]；dH0[j,i] = Λ_{−1}[j,i]
    六项梯度（dq/dk/dv/dβ/dg/dH0）全实现，F 波已用有限差分 + torch autograd 双独立锚证真
    （`attempts/F/f_gdn_golden.py`：worst 7.28e-10）。g 摊成每键通道后，dg 从 F 件的"每 token
    一个标量"变成"每 token 每键通道一个"，收口式子按上式逐通道走（da_i 先沿 j 归约、再乘 a_i）。

    读写次序是本件的核心纪律（两档不一样，别顺手合并）：
      * `ub`：Λ 与 decay/k/q/v/dO/u/dp 暂存全在 UB（都被访问，避开 G-C8 空句柄）；
        趟 1（j 外）算 uj/dp/dv 并读**未更新**的 Λ；趟 2（i 外）逐元素读 Λ^tot 后**立刻**
        回写 Λ_{t−1}（每元素读一次即写，无二次读 ⇒ 无自噬），dq/dk/dg 在 i 外层用寄存器累加
        ⇒ 单存，零 GM 读改写、零 MTE。
      * `pure`：零 alloc_shared。次序换成 **j 外层单趟融合**——第 j 行的 du/p 只读第 j 行 Λ
        （该行此刻必未被更新），可安全就地 RMW；dq/dk/dg 跨 j 累加走 GM 读改写，首列 j==0
        用 `T.if_then_else` 走"首存"⇒ **不要求宿主预清零**。代价：同一地址跨 token 的
        read-modify-write 可见性上卡才能证真（G-F3）⇒ **首推 ub 档**。

    :param dstate: `zero`=末状态梯度视为 0（截断 BPTT 常规口径，接口 `backward` 走这一档）；
        `gm`=末状态梯度由宿主预置在 `Lam`（跨段续推时用，本波只在件里留着，接口未接）。
    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")）。
    """
    import tilelang.ascend.language as T

    seq = T.dynamic("seq")
    if heads % cores:
        raise ValueError(f"heads={heads} 必须能被 cores={cores} 整除（反向按 head 并行，G-F2）")
    upc = heads // cores
    if dstate not in ("zero", "gm"):
        raise ValueError(f"未知 dstate={dstate!r}，只支持 zero/gm")

    @T.prim_func
    def gdn_bwd_impl(Q: T.Tensor((heads, seq, dk), "float32"),
                     K: T.Tensor((heads, seq, dk), "float32"),
                     V: T.Tensor((heads, seq, dv), "float32"),
                     G: T.Tensor((heads, seq, dk), "float32"),
                     Beta: T.Tensor((heads, seq), "float32"),
                     DO: T.Tensor((heads, seq, dv), "float32"),
                     Hist: T.Tensor((heads, seq, dv, dk), "float32"),
                     Lam: T.Tensor((heads, dv, dk), "float32"),
                     DQ: T.Tensor((heads, seq, dk), "float32"),
                     DKG: T.Tensor((heads, seq, dk), "float32"),
                     DV_: T.Tensor((heads, seq, dv), "float32"),
                     DBeta: T.Tensor((heads, seq), "float32"),
                     DG: T.Tensor((heads, seq, dk), "float32"),
                     DH0: T.Tensor((heads, dv, dk), "float32")):
        """被追踪的那一层：一个核守若干个头的整块伴随态，token 轴逆序串行。

        白话：倒查账的工人一人管几整本账（每本一块 值维×键维 的格子），从最后一页往前翻；
        翻一页先把这一页读出去的那笔账并进格子里，再沿键通道把这一页该改的都算出来。
        
        """
        with T.Kernel(cores) as core_id:
            a_i = T.alloc_var("float32", T.float32(0.0))
            bt = T.alloc_var("float32", T.float32(0.0))
            hp = T.alloc_var("float32", T.float32(0.0))
            sp = T.alloc_var("float32", T.float32(0.0))
            lam = T.alloc_var("float32", T.float32(0.0))
            ki = T.alloc_var("float32", T.float32(0.0))
            qi = T.alloc_var("float32", T.float32(0.0))
            p = T.alloc_var("float32", T.float32(0.0))
            uj = T.alloc_var("float32", T.float32(0.0))
            du = T.alloc_var("float32", T.float32(0.0))
            dp = T.alloc_var("float32", T.float32(0.0))
            dst = T.alloc_var("float32", T.float32(0.0))
            vj = T.alloc_var("float32", T.float32(0.0))
            doj = T.alloc_var("float32", T.float32(0.0))
            dq_acc = T.alloc_var("float32", T.float32(0.0))
            dk_acc = T.alloc_var("float32", T.float32(0.0))
            da_acc = T.alloc_var("float32", T.float32(0.0))
            db_acc = T.alloc_var("float32", T.float32(0.0))

            if variant == "ub":
                lam_ub = T.alloc_shared((dv, dk), "float32")  # 伴随态 Λ（本核独占整块）
                dec_ub = T.alloc_shared((dk,), "float32")     # 每键通道衰减系数 exp(g)
                u_scr = T.alloc_shared((dv,), "float32")      # u_t[j]
                dp_scr = T.alloc_shared((dv,), "float32")     # dp_t[j]
                k_ub = T.alloc_shared((dk,), "float32")
                q_ub = T.alloc_shared((dk,), "float32")
                v_scr = T.alloc_shared((dv,), "float32")
                do_scr = T.alloc_shared((dv,), "float32")

            with T.Vector():
                for rr in T.serial(upc):
                    h = core_id * upc + rr

                    # ── Λ 初值：末状态梯度（zero=写 0 / gm=读宿主预置的 Lam）──
                    for j in T.serial(dv):
                        for i in T.serial(dk):
                            if variant == "ub":
                                if dstate == "gm":
                                    lam_ub[j, i] = Lam[h, j, i]
                                else:
                                    lam_ub[j, i] = T.float32(0.0)
                            else:
                                if dstate == "zero":
                                    Lam[h, j, i] = T.float32(0.0)

                    for tt in T.serial(seq):
                        t = (seq - 1) - tt                    # 逆序遍历 token
                        bt = Beta[h, t]
                        db_acc = T.float32(0.0)

                        if variant == "ub":
                            for i in T.serial(dk):
                                dec_ub[i] = T.exp(G[h, t, i])  # → codegen expf（compat §11）
                                k_ub[i] = K[h, t, i]
                                q_ub[i] = Q[h, t, i]
                            for j in T.serial(dv):
                                v_scr[j] = V[h, t, j]
                                do_scr[j] = DO[h, t, j]

                            # 趟 1（j 外）：p / u / du / dp / dv / dβ 累加
                            for j in T.serial(dv):
                                p = T.float32(0.0)
                                du = T.float32(0.0)
                                for i in T.serial(dk):
                                    p = p + (dec_ub[i] * Hist[h, t, j, i]) * k_ub[i]
                                    du = du + (lam_ub[j, i] + do_scr[j] * q_ub[i]) * k_ub[i]
                                uj = bt * (v_scr[j] - p)
                                dp = T.float32(0.0) - bt * du
                                u_scr[j] = uj
                                dp_scr[j] = dp
                                DV_[h, t, j] = bt * du
                                db_acc = db_acc + du * (v_scr[j] - p)
                            DBeta[h, t] = db_acc

                            # 趟 2（i 外）：Λ 递推 + dq[i] / dk[i] / dg[i]
                            for i in T.serial(dk):
                                dq_acc = T.float32(0.0)
                                dk_acc = T.float32(0.0)
                                da_acc = T.float32(0.0)
                                for j in T.serial(dv):
                                    # Λ^tot = Λ_{t+1} + dO⊗q（o_t 对 S_t 的直贡献，必须补）
                                    lam = lam_ub[j, i] + do_scr[j] * q_ub[i]
                                    hp = Hist[h, t, j, i]
                                    sp = dec_ub[i] * hp                   # S'_t[j,i]
                                    dq_acc = dq_acc + (sp + u_scr[j] * k_ub[i]) * do_scr[j]
                                    dk_acc = dk_acc + u_scr[j] * lam + sp * dp_scr[j]
                                    dst = lam + k_ub[i] * dp_scr[j]       # dS'_t[j,i]
                                    da_acc = da_acc + hp * dst
                                    lam_ub[j, i] = dec_ub[i] * dst        # Λ_{t−1}[j,i]
                                DQ[h, t, i] = dq_acc
                                DKG[h, t, i] = dk_acc
                                DG[h, t, i] = da_acc * dec_ub[i]

                        else:
                            # pure：**j 外层单趟融合**，第 j 行读完才写该行 ⇒ 就地 RMW 安全
                            for j in T.serial(dv):
                                vj = V[h, t, j]
                                doj = DO[h, t, j]
                                p = T.float32(0.0)
                                du = T.float32(0.0)
                                for i in T.serial(dk):
                                    ki = K[h, t, i]
                                    qi = Q[h, t, i]
                                    sp = T.exp(G[h, t, i]) * Hist[h, t, j, i]
                                    p = p + sp * ki
                                    du = du + (Lam[h, j, i] + doj * qi) * ki
                                uj = bt * (vj - p)
                                dp = T.float32(0.0) - bt * du
                                DV_[h, t, j] = bt * du
                                db_acc = db_acc + du * (vj - p)
                                for i in T.serial(dk):
                                    ki = K[h, t, i]
                                    a_i = T.exp(G[h, t, i])
                                    # Λ^tot = Λ_{t+1} + dO⊗q（本行 Lam 仍是未更新的 Λ_{t+1}）
                                    lam = Lam[h, j, i] + doj * Q[h, t, i]
                                    hp = Hist[h, t, j, i]
                                    sp = a_i * hp
                                    dq_acc = (sp + uj * ki) * doj
                                    dk_acc = uj * lam + sp * dp
                                    dst = lam + ki * dp
                                    DQ[h, t, i] = T.if_then_else(
                                        j == 0, dq_acc, DQ[h, t, i] + dq_acc)
                                    DKG[h, t, i] = T.if_then_else(
                                        j == 0, dk_acc, DKG[h, t, i] + dk_acc)
                                    DG[h, t, i] = T.if_then_else(
                                        j == 0, (hp * dst) * a_i, DG[h, t, i] + (hp * dst) * a_i)
                                    Lam[h, j, i] = a_i * dst              # Λ_{t−1} 就地回写
                            DBeta[h, t] = db_acc

                    # ── 出口：dH0 = Λ_{−1}（接口不带初状态入口，本波只在件里备好）──
                    for j in T.serial(dv):
                        for i in T.serial(dk):
                            if variant == "ub":
                                DH0[h, j, i] = lam_ub[j, i]
                            else:
                                DH0[h, j, i] = Lam[h, j, i]

    return gdn_bwd_impl


def _gate3(g: torch.Tensor, ref_shape: tuple[int, ...], seq: int, dk: int) -> torch.Tensor:
    """把衰减门统一成 (heads, seq, dk)：每键通道原样收、每 token 标量门摊成通道。

    白话：淡掉黑板的比例有两种记法——要么每个键通道各记一个，要么整行只记一个。件里只按
    前一种记账，收到后一种就照抄 dk 份，算出来的账一模一样。
    """
    if tuple(g.shape) == tuple(ref_shape):
        return g.reshape(-1, seq, dk).contiguous()
    return g.reshape(-1, seq).unsqueeze(-1).expand(-1, seq, dk).contiguous()


def fwd_workspaces(kwargs: dict[str, Any], seq: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """前向的三块内部缓冲：初状态（恒零）、逐步历史、末状态（都是转置朝向 (heads, dv, dk)）。

    白话：开工前先备好"空白的初始账"（每头一片 0）、"每步旧账照片"的相册、以及"最后账本"的
    存放格；这三样都属模具内部的事，不从接口递进去。
    """
    heads, dk, dv = kwargs["heads"], kwargs["dk"], kwargs["dv"]
    h0 = torch.zeros((heads, dv, dk), dtype=torch.float32, device=device)
    hist = torch.empty((heads, seq, dv, dk), dtype=torch.float32, device=device)
    sout = torch.zeros((heads, dv, dk), dtype=torch.float32, device=device)
    return h0, hist, sout


def plan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
         beta: torch.Tensor, o: torch.Tensor, target: str | None = None,
         variant: str | None = None, blocks: int | None = None,
         emit_hist: bool | None = None) -> dict[str, Any] | None:
    """判据：四路输入同头同长、键/值宽度是编译期常量、全程 fp32 且连续；不过关回 None。

    硬门槛：q/k 同形；g 允许 (heads, seq, dk)（每键通道）或 (heads, seq)（每 token 标量门，
    入口的 `_gate3` 会先摊成前者）；v/o 同形且与 q 同头同长，键宽 dk 与值宽 dv 可以不同但都得
    是常量；beta 是 (heads, seq) 的 fp32。任一不合就回 None，由入口走普通写法。
    昇腾侧另外挑一次核块数：前向要 `heads*dv % cores == 0`，挑不出整除就退到 1 核。

    白话：先核对每个人的三摞纸是不是同样高、宽度对得上、记法是不是细尺子，不合模具就换手工。
    """
    name = ascend_env.normalize_target(target)
    if q.dim() != 3 or q.shape != k.shape:
        return None
    if tuple(g.shape) not in (tuple(q.shape), tuple(q.shape[:2])):
        return None
    if v.dim() != 3 or v.shape[:2] != q.shape[:2]:
        return None
    if o.shape != v.shape:
        return None
    if beta.shape != q.shape[:2]:
        return None
    for t in (q, k, v, g, beta, o):
        if t.dtype != torch.float32 or not t.is_contiguous():
            return None
    heads, seq, dk = (int(x) for x in q.shape)
    dv = int(v.size(2))
    if seq == 0 or dk == 0 or dv == 0:
        return None
    impl = gdn_asc_impl if name == ascend_env.TARGET_ASCEND else gdn_cpu_impl
    kwargs: dict[str, Any] = {"heads": heads, "dk": dk, "dv": dv}
    key = f"gdn[{name}]|h{heads}k{dk}v{dv}"
    ws: tuple[str, ...] = ()
    if name == ascend_env.TARGET_ASCEND:
        if emit_hist is None:
            emit_hist = (os.environ.get(ENV_EMIT_HIST) or "1").strip().lower() not in ("0", "false", "no")
        cores = _pick_cores(heads * dv, _blocks_limit(blocks))
        kwargs.update({"cores": cores, "variant": _asc_variant(variant), "emit_hist": bool(emit_hist)})
        ws = ("H0", "Hist", "Sout")
        key += f"|c{cores}{kwargs['variant']}{'h' if emit_hist else ''}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "seq": seq, "ws": ws}


def plan_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor, o: torch.Tensor, target: str | None = None,
                  variant: str | None = None, blocks: int | None = None,
                  dstate: str = "zero") -> dict[str, Any] | None:
    """反向模具判据：在 `plan` 的契约上再加两条——核数按"head 均分"挑，且前向必须落 Hist。

    白话：倒查账要先有每步的旧账照片，所以反向这一路只认带历史的那份前向模具；照片没落，
    反向模具就不该开（宁可回退手工算，也别交一份看着像的假账）。CPU 那侧反向没有模具
    （语义尺子路走 torch），原样把前向 spec 交回去，由入口落回自动微分。
    """
    fwd = plan(q, k, v, g, beta, o, target, variant=variant, blocks=blocks, emit_hist=True)
    if fwd is None or fwd["target"] != ascend_env.TARGET_ASCEND:
        return fwd
    heads, dk, dv = fwd["kwargs"]["heads"], fwd["kwargs"]["dk"], fwd["kwargs"]["dv"]
    cores = _pick_cores(heads, _blocks_limit(blocks))          # 反向按 head 并行（G-F2）
    if heads % cores:
        return None                                            # 挑不出均分就不开模（理论上到不了）
    var = _asc_variant(variant)
    kwargs = {"heads": heads, "dk": dk, "dv": dv, "cores": cores, "variant": var, "dstate": dstate}
    key = f"gdnb[{fwd['target']}]|h{heads}k{dk}v{dv}|c{cores}{var}{'z' if dstate == 'zero' else 'g'}"
    return {"target": fwd["target"], "key": key, "kwargs": kwargs, "impl": gdn_bwd_asc_impl,
            "seq": fwd["seq"], "ws": ("Hist", "Lam"), "fwd": fwd}


def _compiled_for(spec: dict[str, Any], args: tuple[torch.Tensor, ...]):
    """按 spec 取（或首编）模具；args 是这一趟要递给模具的张量清单（只为描形状，不搬数据）。"""
    return ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](*args, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
        beta: torch.Tensor, o: torch.Tensor, spec: dict[str, Any]) -> bool:
    """取（或首编）内核并真跑一次递推；False 表示模具没开成，由入口层回退（不许假装算过）。

    缓存键不含 seq：递推长度是动态符号，同一份产物服务任意长度（与其余件同一口径）。
    昇腾那侧的工作缓冲（初状态 H0 恒零、历史 Hist、末状态 Sout）在这里现领现用——接口对外
    的张量清单不变（还是 q/k/v/g/beta/o 那六路），H0/末状态属内部事。

    白话：模具只在第一次开，之后这条链子有多长都照同一个走，不必因为换了批长度
    就重新开一次模。
    """
    if spec.get("ws"):
        h0, hist, sout = fwd_workspaces(spec["kwargs"], int(q.size(1)), q.device)
        args = (q, k, v, g, beta, h0, hist, sout, o)
    else:
        args = (q, k, v, g, beta, o)
    kernel = _compiled_for(spec, args)
    if kernel is None:
        return False
    kernel(*args)
    return True


def run_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                 beta: torch.Tensor, dout: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor,
                 dv: torch.Tensor, dg: torch.Tensor, dbeta: torch.Tensor, dh0: torch.Tensor,
                 spec: dict[str, Any]) -> bool:
    """反向两趟：先用前向模具重算一遍并落 Hist，再用伴随模具写 dq/dk/dv/dg/dβ/dH0。

    False = 模具没开成（任一趟），由入口回退到 torch 自动微分。`spec["kwargs"]` 里
    `dstate="zero"` 表示末状态梯度按截断 BPTT 视为 0；`dh0` 仍会被写出（= Λ_{−1}），
    调用方要不要是另一回事。注意这两趟共用同一块 Hist（反向要吃前向刚拍下的照片），
    所以这里不能借道 `run()`——它自己现领缓冲，照片就对不上号了。

    白话：先把这本账正着再走一遍、每一步拍张旧账照片，然后倒着翻这些照片算账。
    """
    fwd = spec["fwd"]
    heads, dkw, dvw = fwd["kwargs"]["heads"], fwd["kwargs"]["dk"], fwd["kwargs"]["dv"]
    seq = int(q.size(1))
    h0, hist, sout = fwd_workspaces(fwd["kwargs"], seq, q.device)
    o_scr = torch.empty((heads, seq, dvw), dtype=torch.float32, device=q.device)
    fwd_args = (q, k, v, g, beta, h0, hist, sout, o_scr)
    fwd_kernel = _compiled_for(fwd, fwd_args)
    if fwd_kernel is None:
        return False
    fwd_kernel(*fwd_args)
    lam = torch.zeros((heads, dvw, dkw), dtype=torch.float32, device=q.device)
    args = (q, k, v, g, beta, dout, hist, lam, dq, dk, dv, dbeta, dg, dh0)
    kernel = _compiled_for(spec, args)
    if kernel is None:
        return False
    kernel(*args)
    return True


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
            beta: torch.Tensor, out_dtype: torch.dtype = torch.float16,
            target: str | None = None, **spec_opts: Any) -> torch.Tensor:
    """前向递推：交回 (..., heads, seq, dv) 的输出；内核不可用时用同口径的普通写法。

    白话：按时间一条条把"淡掉—召回—补写—读出"走完，交回每一步读出来的结果。

    :param spec_opts: 透传给 `plan` 的档位开关（variant/blocks/emit_hist），不传按 env 与缺省。
    :raises ValueError: 头数/长度不一致或键宽值宽对不上时抛出。
    """
    _check(q, k, v, g, beta)
    lead = q.shape[:-2]
    seq, dkw, dv = int(q.size(-2)), int(q.size(-1)), int(v.size(-1))
    # 有效头轴 = 前面所有轴的乘积（B,H 与单批 H 都按同一条路走），形状还原时再拆回去
    q3, k3, v3 = (t.reshape(-1, seq, s).contiguous() for t, s in ((q, dkw), (k, dkw), (v, dv)))
    g3 = _gate3(g, tuple(q.shape), seq, dkw)
    b3 = beta.reshape(-1, seq).contiguous()
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, q.device) == ascend_env.TILELANG:
        a3, b3f, c3, d3 = q3.float(), k3.float(), v3.float(), g3.float()
        bb = b3.float()
        out = torch.empty((int(a3.size(0)), seq, dv), dtype=torch.float32, device=q.device)
        spec = plan(a3, b3f, c3, d3, bb, out, name, **spec_opts)
        if spec is not None and run(a3, b3f, c3, d3, bb, out, spec):
            return out.to(out_dtype).reshape(*lead, seq, dv)
    acc = _eager(q3.float(), k3.float(), v3.float(), g3.float(), b3.float())
    return acc.to(out_dtype).reshape(*lead, seq, dv)


def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
             beta: torch.Tensor, dout: torch.Tensor, target: str | None = None,
             return_dh0: bool = False, **spec_opts: Any) -> tuple[torch.Tensor, ...]:
    """反向：ascend target 走 910B 标量面伴随件（六梯全实现），其余落 torch 自动微分。

    白话：模具开着就用模具倒着翻这本账（dq/dk/dv/dg/dβ，外加一个 dH0）；模具开不动就把前向
    那本账交给通用引擎倒着走一遍——口径与前向完全一致，只是速度不是内核速度。

    与 P1-4 之前的差别：`BWD_STATUS` 不再是 partial，ascend 那一路已经接上内核。torch 那一路
    仍然如实登记 blocker（cpu target 或内核没编成时），因为"这一趟没走模具"是机器可见的事实。

    :param return_dh0: True 时末尾多交回一个初状态梯度 dH0（(heads, dv, dk) 的**转置**朝向；
        本接口初状态恒零，所以默认不交，留这一开关给跨段续推的调用方）。
    :returns: (dq, dk, dv, dg, dbeta) 五个与输入同形的 fp32 梯度（顺序见 `BWD_GRADES`）。
    """
    _check(q, k, v, g, beta)
    if dout.shape != v.shape:
        raise ValueError(f"dout 需与 v 同形 {tuple(v.shape)}，实得 {tuple(dout.shape)}")
    lead = q.shape[:-2]
    seq, dkw, dv = int(q.size(-2)), int(q.size(-1)), int(v.size(-1))
    q3, k3, v3 = (t.reshape(-1, seq, s).contiguous() for t, s in ((q, dkw), (k, dkw), (v, dv)))
    g3 = _gate3(g, tuple(q.shape), seq, dkw)
    b3 = beta.reshape(-1, seq).contiguous()
    dy3 = dout.reshape(-1, seq, dv).contiguous()
    heads = int(q3.size(0))
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, q.device) == ascend_env.TILELANG:
        a3, b3f, c3, d3 = q3.float(), k3.float(), v3.float(), g3.float()
        bb, dyf = b3.float(), dy3.float()
        o_scr = torch.empty((heads, seq, dv), dtype=torch.float32, device=q.device)
        spec = plan_backward(a3, b3f, c3, d3, bb, o_scr, name, **spec_opts)
        if spec is not None and spec.get("fwd") is not None:
            gq = torch.empty_like(a3)
            gk = torch.empty_like(a3)
            gv = torch.empty_like(c3)
            gg = torch.empty_like(a3)
            gb = torch.empty_like(bb)
            gh0 = torch.empty((heads, dv, dkw), dtype=torch.float32, device=q.device)
            if run_backward(a3, b3f, c3, d3, bb, dyf, gq, gk, gv, gg, gb, gh0, spec):
                grads = (gq.reshape(*lead, seq, dkw), gk.reshape(*lead, seq, dkw),
                         gv.reshape(*lead, seq, dv), gg.reshape(*lead, seq, dkw),
                         gb.reshape(*lead, seq))
                return grads + ((gh0.reshape(*lead, dv, dkw),) if return_dh0 else ())
    ascend_env.record_blocker(
        f"gdn_backward[{name}]",
        "本趟走 torch 自动微分（cpu target 的语义尺子路，或昇腾伴随模具未编成）；"
        "昇腾形态见 gdn_asc.gdn_bwd_asc_impl（attempts/F 已证 compile + CPU golden）")
    leaves = [t.detach().clone().float().requires_grad_(True) for t in (q3, k3, v3, g3, b3)]
    out = _eager(*leaves)
    # 注意口径：torch.autograd.grad 只把梯度**返回**给调用方，不会写回 leaf.grad，
    # 所以这里必须用返回值，不能照 backward() 的习惯去读 t.grad（读到的永远是 None）。
    grads = torch.autograd.grad(out, leaves, dy3.float(), retain_graph=False)
    # leaves 的顺序是 (q, k, v, g, beta) ⇒ 交回 (dq, dk, dv, dg, dbeta)
    dq, dk_, dv_, dg_, db_ = (t.contiguous().reshape(*lead, *t.shape[1:]) for t in grads)
    if return_dh0:
        # torch 这条路上初状态恒零且没进计算图 ⇒ dH0 拿不到，如实交回同形的 0（不是漏算）
        dz = torch.zeros(tuple(lead) + (dv, dkw), dtype=torch.float32, device=q.device)
        return (dq, dk_, dv_, dg_, db_, dz)
    return (dq, dk_, dv_, dg_, db_)


def _check(q, k, v, g, beta) -> None:
    """入口契约：q/k 同形、g 与 q 同形或与 q 去掉键维同形、v 与 q 同头同长（值宽可不同）、
    beta 是 (..., heads, seq)、四者同设备。"""
    if q.dim() < 3:
        raise ValueError(f"q 至少三维 (..., heads, seq, 键宽)，实得 {q.dim()} 维")
    if q.shape != k.shape:
        raise ValueError(f"q/k 需同形，实得 {tuple(q.shape)} / {tuple(k.shape)}")
    if tuple(g.shape) not in (tuple(q.shape), tuple(q.shape[:-1])):
        raise ValueError(f"g 需为 {tuple(q.shape)}（每键通道）或 {tuple(q.shape[:-1])}"
                         f"（每 token 标量门），实得 {tuple(g.shape)}")
    if v.dim() < 3 or v.shape[:-1] != q.shape[:-1]:
        raise ValueError(f"v 需与 q 同头同长（只有值宽可不同），实得 {tuple(v.shape)}")
    if beta.shape != q.shape[:-1]:
        raise ValueError(f"beta 需为 {tuple(q.shape[:-1])}，实得 {tuple(beta.shape)}")
    if q.device != v.device or q.device != beta.device:
        raise ValueError("四路输入需在同一设备上")


def _eager(q3: torch.Tensor, k3: torch.Tensor, v3: torch.Tensor, g3: torch.Tensor,
           beta3: torch.Tensor) -> torch.Tensor:
    """回退路径与尺子：把五步递推照写一遍（fp32），与内核逐位同口径。

    白话：没有模具时按同一本账手工走一遍——淡掉、召回、补写、读出，一步都不省略。
    """
    heads, seq, dk = q3.shape
    dv = v3.size(-1)
    out = torch.zeros_like(v3)
    state = torch.zeros(heads, dk, dv, dtype=torch.float32, device=q3.device)
    for t in range(seq):
        state = state * g3[:, t].exp().unsqueeze(-1)      # ① 衰减（每键通道）
        pred = torch.einsum("hcd,hc->hd", state, k3[:, t])  # ② 召回：沿键通道把黑板读一遍
        upd = (v3[:, t] - pred) * beta3[:, t].unsqueeze(-1)   # ③ delta 修正
        state = state + k3[:, t].unsqueeze(-1) * upd.unsqueeze(1)  # ④ 秩一写回
        out[:, t] = torch.einsum("hcd,hc->hd", state, q3[:, t])  # ⑤ 读出：照查询再读一遍
    return out
