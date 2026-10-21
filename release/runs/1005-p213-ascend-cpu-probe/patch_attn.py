from pathlib import Path

p = Path("ascend/kernels/attn_sw_asc.py")
src = p.read_text()

pairs = [
    # 昇腾侧：声明串行用的标量临时量（串行区是单线程的，标量安全）
    ('''            stat_ub = T.alloc_shared((bm, 2), "float32")''',
     '''            stat_ub = T.alloc_shared((bm, 2), "float32")
            blkmax = T.alloc_var("float32")
            vnew = T.alloc_var("float32")
            vcorr = T.alloc_var("float32")
            vsum = T.alloc_var("float32")'''),
    # 权重那一步从 SimtVF 挪回串行：① 不可见格必须显式写 0（哨兵减哨兵=0，指数会成 1）；
    # ② 行内求和用 atomic_add 打在 UB 上既不划算也没有先例，串行累加更稳
    ('''                    with T.SimtVF(threads=VEC_THREADS):
                        # 权重：不可见格显式写 0（哨兵减哨兵等于零，指数就是 1，必须拦）
                        for i, j in T.Parallel(bm, bn):
                            i0 = q_lo + i
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Ps_ub[i, j] = T.exp(Sc_ub[i, j] - stat_ub[i, 0])
                            else:
                                Ps_ub[i, j] = 0.0
                        for i, j in T.Parallel(bm, bn):
                            T.atomic_add(stat_ub[i, 1], Ps_ub[i, j])''',
     '''                    # 第二遍扫本块：算权重并累计分母。仍走串行——不可见格必须显式写 0
                    # （哨兵减哨兵等于 0，指数就成 1 了），而行内求和用原子加打在 UB 上
                    # 既不划算也无先例，串行累加最稳（向量化留待云端 C2）
                    for i in T.serial(bm):
                        i0 = q_lo + i
                        vsum = 0.0
                        for j in T.serial(bn):
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Ps_ub[i, j] = T.exp(Sc_ub[i, j] - stat_ub[i, 0])
                                vsum = vsum + Ps_ub[i, j]
                            else:
                                Ps_ub[i, j] = 0.0
                        stat_ub[i, 1] = stat_ub[i, 1] + vsum'''),
    # 写出那一步：SIMT 区里不能共用标量临时量，改用 T.max 就地兜住除零
    ('''                with T.SimtVF(threads=VEC_THREADS):
                    for i, c in T.Parallel(bm, dim):
                        i0 = q_lo + i
                        if i0 < seq:
                            vrow = stat_ub[i, 1]
                            if vrow < FLOOR:
                                vrow = FLOOR
                            O[hh, i0, c] = Ot_ub[c, i] / vrow''',
     '''                with T.SimtVF(threads=VEC_THREADS):
                    for i, c in T.Parallel(bm, dim):
                        i0 = q_lo + i
                        if i0 < seq:
                            # 并行区里不引共享标量临时量，除零直接就地用 max(分母, 地板) 兜住
                            O[hh, i0, c] = Ot_ub[c, i] / T.max(stat_ub[i, 1], FLOOR)'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:60])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
