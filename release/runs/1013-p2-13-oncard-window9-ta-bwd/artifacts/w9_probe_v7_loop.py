"""w9 probe v5: cumulative statement-level bisection of the gdn_bwd body.

v4 verdict: config plane (workspace auto-alloc, multi-output marshalling) is
innocent — the fault is inside the body. This probe rebuilds the body as
LEVEL-gated cumulative segments (compile-time pruning via Python if on the
ctor param); the first failing level names the offending segment.

  L0: H0->io_s->s_ub->SWS copy pipeline only (copy/cast/2D-store baseline)
  L1: + g_one/b_one [1] loads + tile.exp
  L2: + K load + SWS reload + tile.mul(a_ub, s_ub, g_one[0])
  L3: + t_ub=a*k (2D Parallel with 1-D cross-index) + reduce dim=1
  L4: + V load + u=v-p (1D Parallel elementwise) + tile.mul(u,u,b_one[0])
  L5: + reverse preload leg (LAM zero + SWS/K/Q/V/DO loads + recompute a/p/u)
  L6: + Lam_tot in-place accumulate (lam_ub read-modify + dual 1-D cross-idx)
  L7: + du/dp/dv + DVG store (fp32->fp16 1D cast store)
  L8: + dbeta path (row_ub/reduce/Scope-V GM scalar store)
  L9: + dq path + DQ store
  L10: + dk path + DKG store
  L11: + dg path + lam store + DH0 store
"""
import sys

sys.path.insert(0, "/tmp")
import torch
import torch_npu  # noqa: F401

import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@tilelang.jit(out_idx=[14], pass_configs=pass_configs)
def gb5(B: int, H: int, L: int, DK: int, DV: int, LEVEL: int):
    @T.prim_func
    def main(
        Q: T.Tensor([B, H, L, DK], "float32"),
        K: T.Tensor([B, H, L, DK], "float32"),
        V: T.Tensor([B, H, L, DV], "float32"),
        G: T.Tensor([B, H, L], "float32"),
        Beta: T.Tensor([B, H, L], "float32"),
        H0: T.Tensor([B, H, DV, DK], "float32"),
        DO: T.Tensor([B, H, L, DV], "float32"),
        SWS: T.Tensor([B, H, L, DV, DK], "float32"),
        LAM: T.Tensor([B, H, L + 1, DV, DK], "float32"),
        DQ: T.Tensor([B, H, L, DK], "float32"),
        DKG: T.Tensor([B, H, L, DK], "float32"),
        DVG: T.Tensor([B, H, L, DV], "float32"),
        DBeta: T.Tensor([B, H, L], "float32"),
        DG: T.Tensor([B, H, L], "float32"),
        DH0: T.Tensor([B, H, DV, DK], "float32"),
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as cid:
            b = cid // H
            h = cid % H
            io_k = T.alloc_ub([DK], "float32")
            io_v = T.alloc_ub([DV], "float32")
            io_s = T.alloc_ub([DV, DK], "float32")
            g_one = T.alloc_ub([1], "float32")
            b_one = T.alloc_ub([1], "float32")
            k_ub = T.alloc_ub([DK], "float32")
            q_ub = T.alloc_ub([DK], "float32")
            v_ub = T.alloc_ub([DV], "float32")
            do_ub = T.alloc_ub([DV], "float32")
            pred_ub = T.alloc_ub([DV], "float32")
            u_ub = T.alloc_ub([DV], "float32")
            du_ub = T.alloc_ub([DV], "float32")
            dp_ub = T.alloc_ub([DV], "float32")
            dv_ub = T.alloc_ub([DV], "float32")
            red1_ub = T.alloc_ub([DV], "float32")
            dq_ub = T.alloc_ub([DK], "float32")
            dk_ub = T.alloc_ub([DK], "float32")
            row_ub = T.alloc_ub([1, DV], "float32")
            scol_ub = T.alloc_ub([1], "float32")
            lam2_ub = T.alloc_ub([DV, DK], "float32")  # Lam_tot (no in-place RMW)
            buf2_ub = T.alloc_ub([DV, DK], "float32")  # outer-product scratch
            s_ub = T.alloc_ub([DV, DK], "float32")
            a_ub = T.alloc_ub([DV, DK], "float32")
            t_ub = T.alloc_ub([DV, DK], "float32")
            lam_ub = T.alloc_ub([DV, DK], "float32")

            # L0: copy pipeline baseline
            T.copy(H0[b, h, 0, 0], io_s)
            T.copy(io_s, s_ub)
            T.copy(s_ub, SWS[b, h, 0, 0, 0])

            # L1-L4 in the REAL fwd serial loop (loop-var indexed copies)
            for t in T.serial(L):
                if LEVEL >= 1:
                    T.copy(G[b, h, t:t + 1], g_one)
                    T.tile.exp(g_one, g_one)
                    T.copy(Beta[b, h, t:t + 1], b_one)

                if LEVEL >= 2:
                    T.copy(K[b, h, t, 0], io_k)
                    T.copy(io_k, k_ub)
                    T.copy(SWS[b, h, t, 0, 0], s_ub)
                    T.tile.mul(a_ub, s_ub, g_one[0])

                if LEVEL >= 3:
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = a_ub[j, i] * k_ub[i]
                    T.reduce_sum(t_ub, pred_ub, dim=1)

                if LEVEL >= 4:
                    T.copy(V[b, h, t, 0], io_v)
                    T.copy(io_v, v_ub)
                    for j in T.Parallel(DV):
                        u_ub[j] = v_ub[j] - pred_ub[j]
                    T.tile.mul(u_ub, u_ub, b_one[0])

            # L5: LAM zero (flat) then reverse loop shell
            if LEVEL >= 5:
                for (j, i) in T.Parallel(DV, DK):
                    t_ub[j, i] = 0.0
                T.copy(t_ub, LAM[b, h, L, 0, 0])
            for tt in T.serial(L):
                t = (L - 1) - tt
                if LEVEL >= 5:
                    T.copy(G[b, h, t:t + 1], g_one)
                    T.tile.exp(g_one, g_one)
                    T.copy(Beta[b, h, t:t + 1], b_one)
                    T.copy(LAM[b, h, t + 1, 0, 0], lam_ub)
                    T.copy(SWS[b, h, t, 0, 0], s_ub)
                    T.copy(K[b, h, t, 0], io_k)
                    T.copy(io_k, k_ub)
                    T.copy(Q[b, h, t, 0], io_k)
                    T.copy(io_k, q_ub)
                    T.copy(V[b, h, t, 0], io_v)
                    T.copy(io_v, v_ub)
                    T.copy(DO[b, h, t, 0], io_v)
                    T.copy(io_v, do_ub)
                    T.tile.mul(a_ub, s_ub, g_one[0])
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = a_ub[j, i] * k_ub[i]
                    T.reduce_sum(t_ub, pred_ub, dim=1)
                    for j in T.Parallel(DV):
                        u_ub[j] = v_ub[j] - pred_ub[j]
                    T.tile.mul(u_ub, u_ub, b_one[0])

                # L6 split: (a) 2D-read + single cross-index add; (b) dual
                # cross-index outer-product only; (c) 2D-read x single cross mul
                if LEVEL == 6:
                    for (j, i) in T.Parallel(DV, DK):
                        lam2_ub[j, i] = lam_ub[j, i] + do_ub[j]
                if LEVEL == 7:
                    for (j, i) in T.Parallel(DV, DK):
                        lam2_ub[j, i] = do_ub[j] * q_ub[i]
                if LEVEL == 8:
                    for (j, i) in T.Parallel(DV, DK):
                        lam2_ub[j, i] = lam_ub[j, i] * q_ub[i]
                # L9+ : the SPLIT (fixed) forms, cumulative to the full real body
                if 9 <= LEVEL <= 15:  # Lam_tot split + du
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = do_ub[j] * q_ub[i]
                    for (j, i) in T.Parallel(DV, DK):
                        lam2_ub[j, i] = lam_ub[j, i] + buf2_ub[j, i]
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = lam2_ub[j, i] * k_ub[i]
                    T.reduce_sum(t_ub, du_ub, dim=1)

                if 10 <= LEVEL <= 15:  # dp/dv + DVG store — dp as PURE elementwise
                    # (v11 hypothesis: dp_ub written by tile.mul (L10) then used as a
                    # Parallel SOURCE (L13/L14) trips the fault; write it elementwise)
                    for j in T.Parallel(DV):
                        dp_ub[j] = 0.0 - du_ub[j] * b_one[0]
                    T.tile.mul(dv_ub, du_ub, b_one[0])
                    T.copy(dv_ub, io_v)
                    T.copy(io_v, DVG[b, h, t, 0])

                if 11 <= LEVEL <= 15:  # dbeta
                    for j in T.Parallel(DV):
                        row_ub[0, j] = du_ub[j] * (v_ub[j] - pred_ub[j])
                    T.reduce_sum(row_ub, scol_ub, dim=1)
                    with T.Scope("V"):
                        DBeta[b, h, t] = scol_ub[0]

                if 12 <= LEVEL <= 15:  # dq split
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * k_ub[i]
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = (a_ub[j, i] + buf2_ub[j, i]) * do_ub[j]
                    T.reduce_sum(t_ub, dq_ub, dim=0)
                    T.copy(dq_ub, io_k)
                    T.copy(io_k, DQ[b, h, t, 0])

                if LEVEL == 13:  # dk leg 1 only: one j-bcast mul + reduce dim0 + store
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    T.reduce_sum(buf2_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 14:  # dk leg 2 only: the other j-bcast mul + reduce + store
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = a_ub[j, i] * dp_ub[j]
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 15:  # dk legs 1+2, no add: two muls + reduce + store
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = a_ub[j, i] * dp_ub[j]
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 18:  # dk leg1 ALONE (no L9-L12): bare j-bcast mul + dim0 reduce + store
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    T.reduce_sum(buf2_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 19:  # dk broadcast variant ALONE
                    u2_ub = T.alloc_ub([DV, DK], "float32")
                    T.tile.broadcast(u2_ub, u_ub)
                    T.tile.mul(buf2_ub, lam2_ub, u2_ub)
                    T.reduce_sum(buf2_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])

                if LEVEL == 20:  # dst=t_ub, bare j-bcast: is buf2-as-dst the fault?
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 21:  # minimal compound (parens + 0.0): does compounding immunise?
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = (u_ub[j] * lam2_ub[j, i]) + 0.0
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 22:  # bare j-bcast into buf2, but store DQ instead of DKG
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    T.reduce_sum(buf2_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DQ[b, h, t, 0])
                if LEVEL == 23:  # i-bcast control with dim0 reduce
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = lam2_ub[j, i] * k_ub[i]
                    T.reduce_sum(buf2_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 24:  # j-bcast with dim1 reduce (math wrong, fault-scope only)
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = u_ub[j] * lam2_ub[j, i]
                    T.reduce_sum(buf2_ub, du_ub, dim=1)
                    T.copy(du_ub, io_v)
                    T.copy(io_v, DVG[b, h, t, 0])
                if LEVEL == 25:  # dq-mirror compound: (2D + 0.0) * 1D[j]
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = (lam2_ub[j, i] + 0.0) * u_ub[j]
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])

                if LEVEL == 26:  # dk via serial-j accumulation (fork-green shape): in-place 1D accum
                    for i in T.Parallel(DK):
                        dk_ub[i] = 0.0
                    for jj in T.serial(DV):
                        for i in T.Parallel(DK):
                            dk_ub[i] = dk_ub[i] + lam2_ub[jj, i] * u_ub[jj]
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])
                if LEVEL == 27:  # serial-j, two-step per row (tmp mul then accum add)
                    for i in T.Parallel(DK):
                        dk_ub[i] = 0.0
                    for jj in T.serial(DV):
                        for i in T.Parallel(DK):
                            dq_ub[i] = lam2_ub[jj, i] * u_ub[jj]
                        for i in T.Parallel(DK):
                            dk_ub[i] = dk_ub[i] + dq_ub[i]
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])

                if LEVEL >= 16:  # full dk via tile.broadcast (official channel)
                    u2_ub = T.alloc_ub([DV, DK], "float32")
                    dp2_ub = T.alloc_ub([DV, DK], "float32")
                    T.tile.broadcast(u2_ub, u_ub)    # [DV] -> [DV, DK]
                    T.tile.broadcast(dp2_ub, dp_ub)  # [DV] -> [DV, DK]
                    T.tile.mul(buf2_ub, lam2_ub, u2_ub)  # 2D x 2D
                    T.tile.mul(t_ub, a_ub, dp2_ub)       # 2D x 2D
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = t_ub[j, i] + buf2_ub[j, i]
                    T.reduce_sum(t_ub, dk_ub, dim=0)
                    T.copy(dk_ub, io_k)
                    T.copy(io_k, DKG[b, h, t, 0])

                if LEVEL >= 17:  # dst split + dg + lam tail (dst outer product PASS form)
                    for (j, i) in T.Parallel(DV, DK):
                        buf2_ub[j, i] = k_ub[i] * dp_ub[j]
                    for (j, i) in T.Parallel(DV, DK):
                        t_ub[j, i] = lam2_ub[j, i] + buf2_ub[j, i]
                    for (j, i) in T.Parallel(DV, DK):
                        a_ub[j, i] = s_ub[j, i] * t_ub[j, i]
                    T.reduce_sum(a_ub, red1_ub, dim=1)
                    for j in T.Parallel(DV):
                        row_ub[0, j] = red1_ub[j]
                    T.reduce_sum(row_ub, scol_ub, dim=1)
                    T.tile.mul(scol_ub, scol_ub, g_one[0])
                    with T.Scope("V"):
                        DG[b, h, t] = scol_ub[0]
                    T.tile.mul(lam_ub, t_ub, g_one[0])
                    T.copy(lam_ub, LAM[b, h, t, 0, 0])

            T.copy(LAM[b, h, 0, 0, 0], lam_ub)
            T.copy(lam_ub, io_s)
            T.copy(io_s, DH0[b, h, 0, 0])

    return main


def main():
    B, H, L, DK, DV = 1, 1, 1, 8, 8
    q = torch.randn(B, H, L, DK).npu()
    k = torch.randn(B, H, L, DK).npu()
    v = torch.randn(B, H, L, DV).npu()
    g = -torch.rand(B, H, L).npu()
    beta = torch.rand(B, H, L).npu()
    h0 = torch.randn(B, H, DV, DK).npu()
    do = torch.randn(B, H, L, DV).npu()
    sws = torch.zeros(B, H, L, DV, DK).npu()
    lam = torch.zeros(B, H, L + 1, DV, DK).npu()
    dkg = torch.zeros(B, H, L, DK).npu()
    dvg = torch.zeros(B, H, L, DV).npu()
    dbeta = torch.zeros(B, H, L).npu()
    dg = torch.zeros(B, H, L).npu()
    dh0 = torch.zeros(B, H, DV, DK).npu()
    for level in list(range(0, 18)) + [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
        try:
            ker = gb5(B, H, L, DK, DV, level)
            outs = ker(q, k, v, g, beta, h0, do, sws, lam, dkg, dvg, dbeta, dg, dh0)
            [o.cpu() for o in outs]
            print(f"V7 L{level} PASS")
        except Exception as e:  # noqa: BLE001
            print(f"V7 L{level} FAIL {type(e).__name__}: {str(e)[:90]}")
    print("V7-DONE")


if __name__ == "__main__":
    main()
