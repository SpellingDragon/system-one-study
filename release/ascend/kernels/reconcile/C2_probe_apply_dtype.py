#!/usr/bin/env python3
"""reconcile/C2_probe_apply_dtype.py — P1-1m fp16 缺口的**定点补丁器**（可复跑、可回滚）。

【做什么】把三件“prim_func 硬编码 float32 而入参可以是非 fp32”的产品位宽盲区补齐：
  · `gemm_asc.py`        `gemm_scalar_body(..., idt, odt)`：A/W/C 按实际位宽声明，fp16 装填缓冲同宽
  · `gemm_bwd_dw_asc.py` `dw_scalar_body(..., idt)`：dY/A 按实际位宽声明，累加与 GW 出口恒 fp32
  · `rope_asc.py`        `rope_asc_impl(..., qdt, adt)`：qkv 与角度表各自按实际位宽声明（生产混宽）
  三件的 `plan/_plan` 位宽白名单收成 **fp32/fp16**，bf16 显式拒（bisheng dav-2201 缺 bf16 标量 cast），
  缓存键追加位宽段（同形状换位宽必得新模具）。
【纪律】`*_cpu_impl` 与 target=cpu 语义路**一字不动**；cast 只插在标量 load/store 面（DMA 不许转
类型）；`idt/odt/qdt/adt` 缺省 "float32" ⇒ fp32 轨一个 cast 都不插、kwargs 与改前同值 ⇒ 生成码逐字
不变（零回归，用 --revert 与 fp32 全档复跑双向证明）。
【用法】
  python3 ascend/kernels/reconcile/C2_probe_apply_dtype.py --check     # 只验锚点唯一性
  python3 ascend/kernels/reconcile/C2_probe_apply_dtype.py            # 应用
  python3 ascend/kernels/reconcile/C2_probe_apply_dtype.py --revert    # 回滚到改前
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)                       # ascend/kernels

# (文件, 旧文本, 新文本) —— 每条锚点必须在全文**恰好出现一次**，否则拒绝改动
EDITS = [
    # ───────────────────────────── gemm_asc.py ─────────────────────────────
    ("gemm_asc.py", """产物并就地写出口，位宽转换不在方言里做（出口一律 fp32）。""",
     """产物并就地写出口。位宽口径（P1-1m dtype 矩阵）：CPU 那份仍恒 fp32 出口；昇腾那份按实际位宽
   参数化（`idt`/`odt`）——cast 只发生在**标量 load/store** 面，DMA 装填缓冲恒与源张量同宽，因为
   昇腾搬运不允许顺带转类型（ascend/analysis/vf_checker：“DMA copies cannot perform type casting”）。"""),

    ("gemm_asc.py", """#: M 轴块高：行块太大会浪费并行度（批大小可能等于 1）""",
     '''#: 昇腾侧内核位宽白名单（P1-1m）：fp32/fp16 已在正文按位宽参数化并本地闭环（reconcile/
#: C2_dtype_matrix.md）。**bf16 不在名单里是有意的**：bisheng 15.0.5 在 dav-2201 vector 面对 bf16
#: 标量转换直接 `fatal error: error in backend: not support bf16 type cast`（取证 attempts/B/
#: b_addln_910b.py:23-24 与 attempts/B/dump_bf16.py），而生产口径本就是 fp16。早先这里收 bf16/fp16
#: 而正文声明 fp32 ⇒ 编译按**声明**位宽出核、非 fp32 张量到发射期才被当 fp32 解读（静默错数据，取证
#: reconcile/I_RESULT.md §⑥-3；host 侧则崩在 cython `_check_buffer_dtype`）。位宽不符现在 plan 期拒。
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名（白名单外的 dtype 进不了这里，plan 已挡）
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: M 轴块高：行块太大会浪费并行度（批大小可能等于 1）'''),

    ("gemm_asc.py", """def gemm_scalar_body(T, A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int,
                     act_mode: int):""",
     """def gemm_scalar_body(T, A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int,
                     act_mode: int, idt: str = "float32", odt: str = "float32"):"""),

    ("gemm_asc.py", """    nk = k // bk   # K 分块数：同样常量（k 已被 plan 判过整除）

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):""",
     """    nk = k // bk   # K 分块数：同样常量（k 已被 plan 判过整除）
    # 位宽参数化（P1-1m）：入参/出口声明跟着实际位宽走，算式恒在 fp32 面上做。缺省 float32 时
    # 下面两个 helper 是**恒等函数**（一个 cast 都不插）⇒ fp32 轨的生成码与参数化之前逐字相同。
    def _ld(e):            # 标量读：idt → fp32 累加面
        return e if idt == "float32" else T.cast(e, "float32")

    def _st(e):            # 标量写：fp32 结算面 → odt 交货
        return e if odt == "float32" else T.cast(e, odt)

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), idt), W: T.Tensor((n, k), idt),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), odt)):"""),

    ("gemm_asc.py", """            a_ub = _alloc_tile(T, (bm, bk), "float32")
            w_ub = _alloc_tile(T, (bn, bk), "float32")""",
     """            a_ub = _alloc_tile(T, (bm, bk), "float32")   # 夹址标量填装面：恒 fp32（读时才升宽）
            w_ub = _alloc_tile(T, (bn, bk), idt)           # DMA 装填：必须与源张量同宽（搬运不许转类型）"""),

    ("gemm_asc.py", """                            a_ub[i, kk] = T.if_then_else(row < m, A[rc, ko * bk + kk],""",
     """                            a_ub[i, kk] = T.if_then_else(row < m, _ld(A[rc, ko * bk + kk]),"""),

    ("gemm_asc.py", """                                acc = acc + a_ub[i, kk] * w_ub[j, kk]""",
     """                                acc = acc + a_ub[i, kk] * _ld(w_ub[j, kk])"""),

    ("gemm_asc.py", """                            C[row, mx * bn + j] = v""",
     """                            C[row, mx * bn + j] = _st(v)"""),

    ("gemm_asc.py", """def gemm_asc_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int):""",
     """def gemm_asc_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int,
                idt: str = "float32", odt: str = "float32"):"""),

    ("gemm_asc.py", """    return gemm_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), A, W, bias, C,
                            n=n, k=k, bm=bm, bn=bn, bk=bk, act_mode=act_mode)""",
     """    return gemm_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), A, W, bias, C,
                            n=n, k=k, bm=bm, bn=bn, bk=bk, act_mode=act_mode, idt=idt, odt=odt)"""),

    ("gemm_asc.py", """    硬条件三条：截断口径在支持表里；N/K 都能被某个候选块宽整除；位宽组合在该 target 的白名单里
    （CPU 侧只收 fp32，昇腾侧收 fp32/bf16/fp16 同宽操作数 + fp32 偏置与出口）。任一不满足就回""",
     """    硬条件三条：截断口径在支持表里；N/K 都能被某个候选块宽整除；位宽组合在该 target 的白名单里
    （CPU 侧只收 fp32；昇腾侧收 **fp32/fp16** 同宽操作数与出口 + fp32 偏置，bf16 显式不收，理由见
    `_ASC_DTYPES`）。任一不满足就回"""),

    ("gemm_asc.py", """    if name == ascend_env.TARGET_ASCEND and A.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None""",
     """    if name == ascend_env.TARGET_ASCEND and A.dtype not in _ASC_DTYPES:
        return None"""),

    ("gemm_asc.py", """    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
    impl = gemm_asc_impl if name == ascend_env.TARGET_ASCEND else gemm_cpu_impl
    key = f"gemm[{name}]|n{n}k{k}b{BLOCK_M}x{bn}x{bk}a{act}\"""",
     """    dt = _TL_DT.get(A.dtype, "float32")   # plan 已判 C.dtype == A.dtype 且位宽在白名单内 ⇒ 单段位宽
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
    impl = gemm_asc_impl if name == ascend_env.TARGET_ASCEND else gemm_cpu_impl
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(idt=dt, odt=dt)
    # 键尾的位宽段是防“陈旧模具复用”：同 (n,k,b,act) 换位宽必须另开一份产物，否则 fp32 的产物
    # 会被 fp16 形状命中缓存、发射期把 fp16 内存当 fp32 读（静默错数据）。
    key = f"gemm[{name}]|n{n}k{k}b{BLOCK_M}x{bn}x{bk}a{act}d{dt}\""""),

    # ─────────────────────────── gemm_bwd_dw_asc.py ───────────────────────────
    ("gemm_bwd_dw_asc.py", """④ 累加器 (bn,bk) 与出口都 fp32，降位由入口层负责；⑤ 尾块""",
     """④ 累加器 (bn,bk) 与出口 GW 恒 fp32（归约不降位），交货降位由入口层负责；入参位宽由 `idt`
   参数化（fp32/fp16，P1-1m），cast 只发生在标量 load 处（填装走标量、不走 DMA，故缓冲可留 fp32）；⑤ 尾块"""),

    ("gemm_bwd_dw_asc.py", """#: token 轴块长：与前向 BLOCK_M 同量级，太小喂不饱并行、太大尾块浪费""",
     '''#: 昇腾侧内核位宽白名单（P1-1m）：fp32/fp16；**bf16 显式不收**（bisheng dav-2201 vector 面缺
#: bf16 标量 cast，取证 attempts/B/b_addln_910b.py:23-24 + dump_bf16.py；口径同 gemm_asc._ASC_DTYPES）
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: token 轴块长：与前向 BLOCK_M 同量级，太小喂不饱并行、太大尾块浪费'''),

    ("gemm_bwd_dw_asc.py", """def dw_scalar_body(T, dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):""",
     """def dw_scalar_body(T, dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int,
                     idt: str = "float32"):"""),

    ("gemm_bwd_dw_asc.py", """    iters = (tiles + blocks - 1) // blocks

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), "float32"), A: T.Tensor((m, k), "float32"),
                GW: T.Tensor((n, k), "float32")):""",
     """    iters = (tiles + blocks - 1) // blocks
    # 位宽参数化（P1-1m）：入参按实际位宽声明、累加面与出口 GW 恒 fp32（归约不降位）。缺省
    # float32 时 `_ld` 是恒等函数 ⇒ fp32 轨生成码与参数化之前逐字相同（零回归）。
    def _ld(e):            # 标量读：idt → fp32 累加面
        return e if idt == "float32" else T.cast(e, "float32")

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), idt), A: T.Tensor((m, k), idt),
                GW: T.Tensor((n, k), "float32")):"""),

    ("gemm_bwd_dw_asc.py", """                            dy_ub[tt, i] = T.if_then_else(row < m, dY[rc, iy * bn + i],""",
     """                            dy_ub[tt, i] = T.if_then_else(row < m, _ld(dY[rc, iy * bn + i]),"""),

    ("gemm_bwd_dw_asc.py", """                            a_ub[tt, j] = T.if_then_else(row < m, A[rc, ix * bk + j],""",
     """                            a_ub[tt, j] = T.if_then_else(row < m, _ld(A[rc, ix * bk + j]),"""),

    ("gemm_bwd_dw_asc.py", """def dw_asc_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):""",
     """def dw_asc_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int, idt: str = "float32"):"""),

    ("gemm_bwd_dw_asc.py", """    return dw_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), dY, A, GW,
                          n=n, k=k, tc=tc, bn=bn, bk=bk)""",
     """    return dw_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), dY, A, GW,
                          n=n, k=k, tc=tc, bn=bn, bk=bk, idt=idt)"""),

    ("gemm_bwd_dw_asc.py", """    elif dY.dtype != A.dtype or dY.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None""",
     """    elif dY.dtype != A.dtype or dY.dtype not in _ASC_DTYPES:
        return None"""),

    ("gemm_bwd_dw_asc.py", """    kwargs = {"n": n, "k": k, "tc": BLOCK_TC, "bn": bn, "bk": bk}
    impl = dw_asc_impl if name == ascend_env.TARGET_ASCEND else dw_cpu_impl
    key = f"gemm_dw[{name}]|n{n}k{k}tc{BLOCK_TC}b{bn}x{bk}\"""",
     """    dt = _TL_DT.get(dY.dtype, "float32")   # plan 已判 A 与 dY 同宽、GW 恒 fp32、位宽在白名单内
    kwargs = {"n": n, "k": k, "tc": BLOCK_TC, "bn": bn, "bk": bk}
    impl = dw_asc_impl if name == ascend_env.TARGET_ASCEND else dw_cpu_impl
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(idt=dt)
    # 键尾位宽段：同 (n,k,块) 换位宽必得新模具（防陈旧产物把 fp16 内存当 fp32 读，见 gemm_asc 同注）
    key = f"gemm_dw[{name}]|n{n}k{k}tc{BLOCK_TC}b{bn}x{bk}d{dt}\""""),

    # ────────────────────────────── rope_asc.py ──────────────────────────────
    ("rope_asc.py", """#: 打包布局里被旋转的槽位：0=查询、1=键；2=值 永远不动（与 P1 rope_mps.ROTATE_SLOTS 同值）""",
     '''#: 昇腾侧内核位宽白名单（P1-1m）：qkv 与角度表都只收 fp32/fp16；**bf16 显式不收**（bisheng
#: dav-2201 vector 面缺 bf16 标量 cast，取证 attempts/B/b_addln_910b.py:23-24 + dump_bf16.py）。
#: 早先 _plan 收 bf16/fp16 而正文声明 fp32 ⇒ 编译按声明位宽出核、非 fp32 张量到发射期才被当 fp32
#: 解读（静默错数据，取证 reconcile/I_RESULT.md §⑥-3）；现在位宽在 plan 期就拒或按 dtype 参数化。
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: 打包布局里被旋转的槽位：0=查询、1=键；2=值 永远不动（与 P1 rope_mps.ROTATE_SLOTS 同值）'''),

    ("rope_asc.py", """def rope_asc_impl(QKV, Cos, Sin, heads: int, dim: int, slots: int, sign: int):""",
     """def rope_asc_impl(QKV, Cos, Sin, heads: int, dim: int, slots: int, sign: int,
                qdt: str = "float32", adt: str = "float32"):"""),

    ("rope_asc.py", """    @T.prim_func
    def rope_impl(QKV: T.Tensor((tokens, 3, heads, dim), "float32"),
                  Cos: T.Tensor((tokens, half), "float32"),
                  Sin: T.Tensor((tokens, half), "float32")):
        \"\"\"被追踪的那一层：头数、每头宽度、转几摞、正转还是反转都是常量，只有条数是活的。""",
     """    # 位宽参数化（P1-1m）：qkv 与两张角度表各自按实际位宽声明（生产就是 qkv fp16 + 表 fp32 的
    # 混宽），**计算量仍恒 fp32**（alloc_var 不动，只在标量 load/store 处 cast）——与 `_eager`
    # “fp32 累加、按 qkv 位宽交货”同口径。三个 helper 在位宽为 float32 时都是恒等函数 ⇒
    # fp32 轨的生成码与参数化之前逐字相同（零回归）。
    def _lq(e):            # 标量读 qkv：qdt → fp32
        return e if qdt == "float32" else T.cast(e, "float32")

    def _la(e):            # 标量读角度表：adt → fp32
        return e if adt == "float32" else T.cast(e, "float32")

    def _sq(e):            # 标量写 qkv：fp32 结算 → qdt（就地语义、不新增全局缓冲）
        return e if qdt == "float32" else T.cast(e, qdt)

    @T.prim_func
    def rope_impl(QKV: T.Tensor((tokens, 3, heads, dim), qdt),
                  Cos: T.Tensor((tokens, half), adt),
                  Sin: T.Tensor((tokens, half), adt)):
        \"\"\"被追踪的那一层：头数、每头宽度、转几摞、正转还是反转都是常量，只有条数是活的。"""),

    ("rope_asc.py", """                                x1 = QKV[t, s, hh, j]
                                x2 = QKV[t, s, hh, j + half]
                                c = Cos[t, j]
                                sn = Sin[t, j]
                                if sign < 0:
                                    sn = T.float32(0.0) - sn
                                QKV[t, s, hh, j] = x1 * c - x2 * sn
                                QKV[t, s, hh, j + half] = x2 * c + x1 * sn""",
     """                                x1 = _lq(QKV[t, s, hh, j])
                                x2 = _lq(QKV[t, s, hh, j + half])
                                c = _la(Cos[t, j])
                                sn = _la(Sin[t, j])
                                if sign < 0:
                                    sn = T.float32(0.0) - sn
                                QKV[t, s, hh, j] = _sq(x1 * c - x2 * sn)
                                QKV[t, s, hh, j + half] = _sq(x2 * c + x1 * sn)"""),

    ("rope_asc.py", """    elif qkv.dtype not in (torch.float32, torch.bfloat16, torch.float16) or qkv.dtype != cos.dtype:
        return None""",
     """    else:
        # 昇腾面位宽判据（P1-1m，单点裁决）：qkv 与两张表都只收 fp32/fp16；两张表必须同宽
        # （sin 也要比——早先只比 cos，sin 换宽就是静默错数据）；qkv 与表**允许不同宽**：生产的
        # cos/sin 出自 train_step._build_params 的 torch.cos(angle) ⇒ fp32，而主干 qkv 是 fp16，
        # 早先那条 `qkv.dtype != cos.dtype` 会把生产形状直接判死、内核永不被调用。
        if (qkv.dtype not in _ASC_DTYPES or cos.dtype not in _ASC_DTYPES
                or sin.dtype not in _ASC_DTYPES or cos.dtype != sin.dtype):
            return None"""),

    ("rope_asc.py", """    kwargs = {"heads": heads, "dim": dim, "slots": slots, "sign": sign}
    impl = rope_asc_impl if name == ascend_env.TARGET_ASCEND else rope_cpu_impl
    key = f"rope[{name}]|h{heads}d{dim}s{slots}g{sign}\"""",
     """    kwargs = {"heads": heads, "dim": dim, "slots": slots, "sign": sign}
    impl = rope_asc_impl if name == ascend_env.TARGET_ASCEND else rope_cpu_impl
    qdt, adt = _TL_DT.get(qkv.dtype, "float32"), _TL_DT.get(cos.dtype, "float32")
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(qdt=qdt, adt=adt)
    # 键尾位宽段：同 (h,dim,slots,sign) 换位宽必得新模具（防陈旧产物错位读内存，见 gemm_asc 同注）
    key = f"rope[{name}]|h{heads}d{dim}s{slots}g{sign}q{qdt}a{adt}\""""),

    ("rope_asc.py", """    elif (name == ascend_env.TARGET_ASCEND
          and ascend_env.active_backend(name, qkv.device) == ascend_env.TILELANG
          and qkv.dtype == torch.float32 and cos.dtype == torch.float32 and sin.dtype == torch.float32):
        # P1-4 合流（路由修复）：target=ascend 现走内核；此前 forward 无 ascend 分支，
        # 卡上会永落 _eager（内核白编）。限 fp32：ascend 正文声明 fp32，非 fp32 落 _eager 以免
        # 静默错数据（编译门对位宽错配是瞎的，见 D-int 侦察）；fp16 rope 内核化列为后续。""",
     """    elif (name == ascend_env.TARGET_ASCEND
          and ascend_env.active_backend(name, qkv.device) == ascend_env.TILELANG):
        # P1-4 合流（路由修复）：target=ascend 现走内核；此前 forward 无 ascend 分支，
        # 卡上会永落 _eager（内核白编）。P1-1m（dtype 矩阵）：位宽判据从这里的“硬编码 fp32 三门”
        # 下放到 `_plan` **单点裁决**——正文已按 qdt/adt 参数化（fp32 累加、按实际位宽交货），
        # fp16 主干＋fp32 角度表就是生产形状；白名单外的位宽（bf16 等）由 _plan 回 None，照旧落
        # _eager，不会出现“按声明位宽出核、把 fp16 内存当 fp32 读”的静默错数据。"""),
]


def read(fn):
    with io.open(os.path.join(KDIR, fn), encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    texts, miss = {}, []
    for fn, old, new in EDITS:
        src = texts.get(fn, read(fn))
        a, b = (new, old) if args.revert else (old, new)
        cnt = src.count(a)
        if cnt != 1:                                  # 锚点缺失或不唯一 ⇒ 整批不动
            miss.append((fn, cnt, a.splitlines()[0][:70]))
            continue
        texts[fn] = src.replace(a, b, 1)
    if miss:
        print(f"锚点异常 {len(miss)} 处（未做任何改动）：")
        for fn, cnt, head in miss:
            print(f"  - {fn} 命中 {cnt} 次: {head}")
        return 2
    if args.check:
        print(f"锚点全部唯一、可替换 {len(EDITS)} 处，涉及 {sorted(set(f for f, _, _ in EDITS))}")
        return 0
    for fn, src in sorted(texts.items()):
        with io.open(os.path.join(KDIR, fn), "w", encoding="utf-8") as f:
            f.write(src)
        print(f"{'回滚' if args.revert else '应用'}: {fn}（{len(src.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
