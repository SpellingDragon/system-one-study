from pathlib import Path

p = Path("ascend/kernels/add_ln_asc.py")
src = p.read_text()

# A. CPU 反向：dy 单独一份 tile，列梯度用原始 dy，不再用 w/G 反推（G 可能为 0）
pairs = [
    ("""            h = T.alloc_local((bm, dim), "float32")
            w = T.alloc_local((bm, dim), "float32")
            xh = T.alloc_local((bm, dim), "float32")""",
     """            h = T.alloc_local((bm, dim), "float32")
            dy = T.alloc_local((bm, dim), "float32")
            w = T.alloc_local((bm, dim), "float32")
            xh = T.alloc_local((bm, dim), "float32")"""),
    ('                T.copy(DY[blk * bm, 0], w)', '                T.copy(DY[blk * bm, 0], dy)'),
    ("""                        for j in T.serial(dim):
                            xh[r, j] = xh[r, j] * rs
                            w[r, j] = w[r, j] * G[j]""",
     """                        for j in T.serial(dim):
                            xh[r, j] = xh[r, j] * rs
                            w[r, j] = dy[r, j] * G[j]"""),
    ("""                            # 列梯度是"往上加一笔"，所以入口层必须先把它清零
                            Dg[j] = Dg[j] + w[r, j] / G[j] * xh[r, j]
                            Db[j] = Db[j] + w[r, j] / G[j]""",
     """                            # 列梯度用原始 dy（倍率还没乘上去的那份），且是"加一笔"，
                            # 所以入口层必须先把它清零
                            Dg[j] = Dg[j] + dy[r, j] * xh[r, j]
                            Db[j] = Db[j] + dy[r, j]"""),
    # B. 昇腾反向正文：均值/尺度/两个修正均值各占独立槽位，dx 用尺度乘括号项；列梯度原子加
    ("""                        h_fr = T.alloc_fragment((bm, dim), "float32")
                        dy_fr = T.alloc_fragment((bm, dim), "float32")
                        w_fr = T.alloc_fragment((bm, dim), "float32")
                        xh_fr = T.alloc_fragment((bm, dim), "float32")
                        sq_fr = T.alloc_fragment((bm, dim), "float32")
                        wd_fr = T.alloc_fragment((bm, dim), "float32")
                        dx_fr = T.alloc_fragment((bm, dim), "float32")
                        s_fr = T.alloc_fragment((bm,), "float32")
                        v_fr = T.alloc_fragment((bm,), "float32")
                        m1_fr = T.alloc_fragment((bm,), "float32")
                        m2_fr = T.alloc_fragment((bm,), "float32")
                        g_fr = T.alloc_fragment((dim,), "float32")
                        T.copy(h_ub, h_fr)
                        T.copy(dy_ub, dy_fr)
                        T.copy(g_ub, g_fr)
                        # 复算前向的两个统计量（与前向完全同式，杜绝"存下来再回读"）
                        T.reduce_sum(h_fr, s_fr, dim=1)
                        for i in T.Parallel(bm):
                            m1_fr[i] = s_fr[i] / dim          # 这里先当均值用
                        for i, j in T.Parallel(bm, dim):
                            sq_fr[i, j] = h_fr[i, j] - m1_fr[i]
                            w_fr[i, j] = sq_fr[i, j] * sq_fr[i, j]
                        T.reduce_sum(w_fr, v_fr, dim=1)
                        for i in T.Parallel(bm):
                            m2_fr[i] = T.rsqrt(v_fr[i] / dim + eps)   # 这里先当尺度用
                        for i, j in T.Parallel(bm, dim):
                            xh_fr[i, j] = sq_fr[i, j] * m2_fr[i]
                            w_fr[i, j] = dy_fr[i, j] * g_fr[j]
                        # 闭式反向的两个均值：mean(wdy) 与 mean(wdy*xhat)
                        T.reduce_sum(w_fr, m1_fr, dim=1)
                        for i in T.Parallel(bm):
                            m1_fr[i] = m1_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            wd_fr[i, j] = w_fr[i, j] * xh_fr[i, j]
                        T.reduce_sum(wd_fr, m2_fr, dim=1)
                        for i in T.Parallel(bm):
                            m2_fr[i] = m2_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            dx_fr[i, j] = m2_fr[i + bm * 0] * 0.0 + (
                                w_fr[i, j] - m1_fr[i] - xh_fr[i, j] * m2_fr[i]) * sq_fr[i, j] * 0.0
                        T.copy(dx_fr, dx_ub)""",
     """                        h_fr = T.alloc_fragment((bm, dim), "float32")
                        dy_fr = T.alloc_fragment((bm, dim), "float32")
                        d_fr = T.alloc_fragment((bm, dim), "float32")
                        sq_fr = T.alloc_fragment((bm, dim), "float32")
                        w_fr = T.alloc_fragment((bm, dim), "float32")
                        xh_fr = T.alloc_fragment((bm, dim), "float32")
                        wd_fr = T.alloc_fragment((bm, dim), "float32")
                        dx_fr = T.alloc_fragment((bm, dim), "float32")
                        s_fr = T.alloc_fragment((bm,), "float32")
                        v_fr = T.alloc_fragment((bm,), "float32")
                        mu_fr = T.alloc_fragment((bm,), "float32")
                        rs_fr = T.alloc_fragment((bm,), "float32")
                        m1_fr = T.alloc_fragment((bm,), "float32")
                        m2_fr = T.alloc_fragment((bm,), "float32")
                        g_fr = T.alloc_fragment((dim,), "float32")
                        T.copy(h_ub, h_fr)
                        T.copy(dy_ub, dy_fr)
                        T.copy(g_ub, g_fr)
                        # 复算前向的两个统计量（与前向完全同式，杜绝"存下来再回读"）
                        T.reduce_sum(h_fr, s_fr, dim=1)
                        for i in T.Parallel(bm):
                            mu_fr[i] = s_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            d_fr[i, j] = h_fr[i, j] - mu_fr[i]
                            sq_fr[i, j] = d_fr[i, j] * d_fr[i, j]
                        T.reduce_sum(sq_fr, v_fr, dim=1)
                        for i in T.Parallel(bm):
                            rs_fr[i] = T.rsqrt(v_fr[i] / dim + eps)
                        for i, j in T.Parallel(bm, dim):
                            xh_fr[i, j] = d_fr[i, j] * rs_fr[i]
                            w_fr[i, j] = dy_fr[i, j] * g_fr[j]
                        # 闭式反向的两个均值：mean(wdy) 与 mean(wdy*xhat)
                        T.reduce_sum(w_fr, m1_fr, dim=1)
                        for i, j in T.Parallel(bm, dim):
                            wd_fr[i, j] = w_fr[i, j] * xh_fr[i, j]
                        T.reduce_sum(wd_fr, m2_fr, dim=1)
                        for i in T.Parallel(bm):
                            m1_fr[i] = m1_fr[i] / dim
                            m2_fr[i] = m2_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            dx_fr[i, j] = rs_fr[i] * (w_fr[i, j] - m1_fr[i] - xh_fr[i, j] * m2_fr[i])
                            # 列方向跨块累计，只能"加一笔"而不是"盖一笔"
                            T.atomic_add(Dg[j], dy_fr[i, j] * xh_fr[i, j])
                            T.atomic_add(Db[j], dy_fr[i, j])
                        T.copy(dx_fr, dx_ub)"""),
    # C. 昇腾侧块宽随 dim 收缩（fragment 摊在寄存器里，宽行必须缩块）
    ("""def _plan_fwd(X, Res, G, Bt, Y, Hout, eps, target):""",
     """def _pick_bm(dim: int, name: str) -> int:
    \"\"\"块宽：CPU 走满 ROW_BLOCK（验证口径），昇腾按 fragment 预算收缩，宽行绝不一上来就摊满。

    白话：手工看一行要看几眼就够，机器把每行摊在随手可及的小格子里，行越宽，一次能摊的行数
    就得越少，否则小格子先满了。
    \"\"\"
    if name != ascend_env.TARGET_ASCEND:
        return ROW_BLOCK
    return max(1, min(ROW_BLOCK, ASCEND_FRAG_BUDGET // max(dim, 1)))


def _plan_fwd(X, Res, G, Bt, Y, Hout, eps, target):"""),
    ('NUM_STAGES = 2\n', 'NUM_STAGES = 2\n#: 昇腾侧每块 (bm, dim) fragment 的元素数预算（宽行缩块用，云端 C1 再按实测 UB 调）\nASCEND_FRAG_BUDGET = 8192\n'),
    ('''    kwargs = {"dim": dim, "eps": float(eps), "bm": ROW_BLOCK}
    impl = add_ln_asc_impl if name == ascend_env.TARGET_ASCEND else add_ln_cpu_impl
    key = f"add_ln[{name}]|d{dim}e{float(eps)}b{ROW_BLOCK}"''',
     '''    bm = _pick_bm(dim, name)
    kwargs = {"dim": dim, "eps": float(eps), "bm": bm}
    impl = add_ln_asc_impl if name == ascend_env.TARGET_ASCEND else add_ln_cpu_impl
    key = f"add_ln[{name}]|d{dim}e{float(eps)}b{bm}"'''),
    ('''    kwargs = {"dim": dim, "eps": float(eps), "bm": ROW_BLOCK}
    impl = ln_bwd_asc_impl if name == ascend_env.TARGET_ASCEND else ln_bwd_cpu_impl
    key = f"ln_bwd[{name}]|d{dim}e{float(eps)}b{ROW_BLOCK}"''',
     '''    bm = _pick_bm(dim, name)
    kwargs = {"dim": dim, "eps": float(eps), "bm": bm}
    impl = ln_bwd_asc_impl if name == ascend_env.TARGET_ASCEND else ln_bwd_cpu_impl
    key = f"ln_bwd[{name}]|d{dim}e{float(eps)}b{bm}"'''),
]
for old, new in pairs:
    assert src.count(old) == 1, ("pattern", src.count(old), old[:60])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
