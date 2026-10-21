from pathlib import Path

p = Path("ascend/kernels/gdn_asc.py")
src = p.read_text()

pairs = [
    # A. 入口的形状还原：多头批（B,H,T,D）必须把 (B,H) 合成"有效头"轴，不能拿 H 当块数
    ('''    _check(q, k, v, g, beta)
    lead = q.shape[:-2]
    dq, dk_, dv, heads = q.size(-2), q.size(-1), v.size(-1), q.size(-3)
    q3, k3, v3, g3 = (t.reshape(heads, int(q.shape[-2]), s).contiguous()
                      for t, s in ((q, dk_), (k, dk_), (v, dv), (g, dk_)))
    b3 = beta.reshape(heads, int(q.shape[-2])).contiguous()
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, q.device) == ascend_env.TILELANG:
        a3, b3f, c3, d3 = q3.float(), k3.float(), v3.float(), g3.float()
        bb, out = b3.float(), torch.empty((heads, int(q.shape[-2]), dv), dtype=torch.float32,
                                          device=q.device)''',
     '''    _check(q, k, v, g, beta)
    lead = q.shape[:-2]
    seq, dk_, dv = int(q.size(-2)), int(q.size(-1)), int(v.size(-1))
    # 有效头轴 = 前面所有轴的乘积（B,H 与单批 H 都按同一条路走），形状还原时再拆回去
    q3, k3, v3, g3 = (t.reshape(-1, seq, s).contiguous() for t, s in ((q, dk_), (k, dk_), (v, dv), (g, dk_)))
    b3 = beta.reshape(-1, seq).contiguous()
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, q.device) == ascend_env.TILELANG:
        a3, b3f, c3, d3 = q3.float(), k3.float(), v3.float(), g3.float()
        bb, out = b3.float(), torch.empty((q3.size(0), seq, dv), dtype=torch.float32, device=q.device)'''),
    ('''        if spec is not None and run(a3, b3f, c3, d3, bb, out, spec):
            return out.to(out_dtype).reshape(*lead, out.size(1), dv)
    acc = _eager(q3.float(), k3.float(), v3.float(), g3.float(), b3.float())
    return acc.to(out_dtype).reshape(*lead, q3.size(1), dv)''',
     '''        if spec is not None and run(a3, b3f, c3, d3, bb, out, spec):
            return out.to(out_dtype).reshape(*lead, seq, dv)
    acc = _eager(q3.float(), k3.float(), v3.float(), g3.float(), b3.float())
    return acc.to(out_dtype).reshape(*lead, seq, dv)'''),
    # B. 尺子里的 einsum 下标写错（键通道与值通道两个标签混用过），改成 hcd,hc->hd
    ('''        pred = torch.einsum("hkd,hdk->hk", state, k3[:, t])  # ② 召回：kᵀ S''',
     '''        pred = torch.einsum("hcd,hc->hd", state, k3[:, t])  # ② 召回：沿键通道把黑板读一遍'''),
    ('''        out[:, t] = torch.einsum("hkd,hdk->hk", state, q3[:, t])  # ⑤ 读出''',
     '''        out[:, t] = torch.einsum("hcd,hc->hd", state, q3[:, t])  # ⑤ 读出：照查询再读一遍'''),
    # C. 昇腾正文：g 也先搬进 UB（并行区不直接读全局），标量临时量在块顶一次声明、别处复用
    ('''            dec_ub = T.alloc_shared((dk,), "float32")
            pred_ub = T.alloc_shared((dv,), "float32")
            upd_ub = T.alloc_shared((dv,), "float32")
            T.annotate_buffer_versions({S: 1})''',
     '''            gt_ub = T.alloc_shared((dk,), "float32")
            dec_ub = T.alloc_shared((dk,), "float32")
            pred_ub = T.alloc_shared((dv,), "float32")
            upd_ub = T.alloc_shared((dv,), "float32")
            acc = T.alloc_var("float32")
            T.annotate_buffer_versions({S: 1})'''),
    ('''                T.copy(V[hh, t, 0], vt_ub)
                with T.SimtVF(threads=VEC_THREADS):
                    for kk in T.Parallel(dk):
                        dec_ub[kk] = T.exp(G[hh, t, kk])''',
     '''                T.copy(V[hh, t, 0], vt_ub)
                T.copy(G[hh, t, 0], gt_ub)
                with T.SimtVF(threads=VEC_THREADS):
                    for kk in T.Parallel(dk):
                        dec_ub[kk] = T.exp(gt_ub[kk])'''),
    ('''                    for kk, vv in T.Parallel(dk, dv):
                        S[kk, vv] = S[kk, vv] * dec_ub[kk]
                    for vv in T.Parallel(dv):
                        # ② 召回：沿键通道的归约先按串行写（向量化留待云端 C2）
                        pred_ub[vv] = 0.0
                bt = Beta[hh, t]
                for vv in T.serial(dv):
                    acc = T.alloc_var("float32")
                    acc = 0.0''',
     '''                    for kk, vv in T.Parallel(dk, dv):
                        S[kk, vv] = S[kk, vv] * dec_ub[kk]
                # ② 召回：沿键通道的归约先按串行写（向量化留待云端 C2）
                bt = Beta[hh, t]
                for vv in T.serial(dv):
                    acc = 0.0'''),
    ('''                for vv in T.serial(dv):
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + qt_ub[kk] * S[kk, vv]
                    O[hh, t, vv] = acc''',
     '''                # ⑤ 读出：写回之后再沿键通道扫一遍
                for vv in T.serial(dv):
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + qt_ub[kk] * S[kk, vv]
                    O[hh, t, vv] = acc'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:60])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
