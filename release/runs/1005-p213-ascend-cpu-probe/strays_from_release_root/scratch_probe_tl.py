"""临时探查 #9：c 后端出口降位宽/就地旋转/LoRA 双 gemm/GDN 递推 四个关键构件。"""
import sys

import torch
import tilelang
import tilelang.cpu.language as T

CFGP = dict(target="c", target_host="c", execution_backend="cython")


def cmp_(prog, out):
    return tilelang.compile(prog, out_idx=out, **CFGP)


def go(name, fn):
    try:
        print(f"[{name}] -> {fn()}", flush=True)
    except Exception as e:
        first = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        msg = first[0][:200] if first else str(e)[:200]
        if "Compilation Failed" in msg:
            err = [ln for ln in str(e).splitlines() if " error:" in ln]
            msg = err[0][:200] if err else msg
        print(f"[{name}] FAIL {type(e).__name__}: {msg}", flush=True)


def ln_fp16_out():
    """fp32 累加 → 逐元素 cast 到 fp16 local → 同位宽 T.copy 出口（绕开 float4→half4 缺陷）。"""
    R, D = 4, 16

    @T.prim_func
    def main(X: T.Tensor((R, D), "float32"), RES: T.Tensor((R, D), "float32"),
             G: T.Tensor((D,), "float32"), BT: T.Tensor((D,), "float32"),
             Y: T.Tensor((R, D), "float16"), H: T.Tensor((R, D), "float32"), eps: T.float32):
        with T.Kernel(R // 2) as bx:
            xl = T.alloc_local((2, D), "float32")
            rl = T.alloc_local((2, D), "float32")
            hl = T.alloc_local((2, D), "float32")
            yl = T.alloc_local((2, D), "float16")
            gl = T.alloc_local((D,), "float32")
            bt = T.alloc_local((D,), "float32")
            mu = T.alloc_local((2,), "float32")
            vr = T.alloc_local((2,), "float32")
            T.copy(X[bx * 2, 0], xl)
            T.copy(RES[bx * 2, 0], rl)
            T.copy(G[0], gl)
            T.copy(BT[0], bt)
            for i in T.serial(2):
                mu[i] = T.float32(0)
                vr[i] = T.float32(0)
                for j in T.serial(D):
                    hl[i, j] = xl[i, j] + rl[i, j]
                    mu[i] = mu[i] + hl[i, j]
            for i in T.serial(2):
                mu[i] = mu[i] / D
                for j in T.serial(D):
                    vr[i] = vr[i] + (hl[i, j] - mu[i]) * (hl[i, j] - mu[i])
            for i in T.serial(2):
                vr[i] = T.rsqrt(vr[i] / D + eps)
                for j in T.serial(D):
                    hl[i, j] = (hl[i, j] - mu[i]) * vr[i] * gl[j] + bt[j]
                    yl[i, j] = T.cast(hl[i, j], "float16")
            T.copy(yl, Y[bx * 2, 0])
            T.copy(hl, H[bx * 2, 0])
    k = cmp_(main, [4, 5])
    x, res, g, bt = torch.rand(4, 16), torch.rand(4, 16), torch.rand(16), torch.rand(16)
    y, h = k(x, res, g, bt, 1e-5)
    href = x + res
    yn = torch.nn.functional.layer_norm(href, (16,), g, bt, 1e-5)
    return "h_err=%.2e y_err=%.2e" % (float((h - href).abs().max()), float((y.float() - yn).abs().max()))


def rope_inplace():
    TOK, H, DIM = 4, 2, 8
    half = DIM // 2

    @T.prim_func
    def main(QKV: T.Tensor((TOK, 3, H, DIM), "float32"), COS: T.Tensor((TOK, half), "float32"),
             SIN: T.Tensor((TOK, half), "float32")):
        with T.Kernel(TOK) as t:
            xl = T.alloc_local((2, H, DIM), "float32")
            cl = T.alloc_local((half,), "float32")
            sl = T.alloc_local((half,), "float32")
            T.copy(QKV[t, 0, 0, 0], xl)
            T.copy(COS[t, 0], cl)
            T.copy(SIN[t, 0], sl)
            for s in T.serial(2):
                for hd in T.serial(H):
                    for i in T.serial(half):
                        x1 = xl[s, hd, i]
                        x2 = xl[s, hd, i + half]
                        xl[s, hd, i] = x1 * cl[i] - x2 * sl[i]
                        xl[s, hd, i + half] = x2 * cl[i] + x1 * sl[i]
            T.copy(xl, QKV[t, 0, 0, 0])
    k = cmp_(main, [])
    qkv = torch.rand(TOK, 3, H, DIM)
    origin = qkv.clone()
    cos, sin = torch.rand(TOK, half), torch.rand(TOK, half)
    k(qkv, cos, sin)
    c = cos.reshape(TOK, 1, 1, half)
    s2 = sin.reshape(TOK, 1, 1, half)
    x1, x2 = origin[:, :2, :, :half], origin[:, :2, :, half:]
    return "rope_err=%.2e vslot_untouched=%s" % (
        float((qkv[:, :2, :, :half] - (x1 * c - x2 * s2)).abs().max()),
        bool(torch.equal(qkv[:, 2], origin[:, 2])))


def lora_side():
    M, N, K, R = 8, 8, 8, 4

    @T.prim_func
    def main(X: T.Tensor((M, K), "float32"), W: T.Tensor((N, K), "float32"), A: T.Tensor((R, K), "float32"),
             B: T.Tensor((R, N), "float32"), C: T.Tensor((M, N), "float32"), s: T.float32):
        with T.Kernel(1) as bx:
            xl = T.alloc_local((M, K), "float32")
            wl = T.alloc_local((N, K), "float32")
            al = T.alloc_local((R, K), "float32")
            bl = T.alloc_local((R, N), "float32")
            hl = T.alloc_local((M, R), "float32")
            cl = T.alloc_local((M, N), "float32")
            T.copy(X[0, 0], xl)
            T.copy(W[0, 0], wl)
            T.copy(A[0, 0], al)
            T.copy(B[0, 0], bl)
            T.clear(cl)
            T.gemm(xl, wl, cl, transpose_B=True, clear_accum=True)
            T.gemm(xl, al, hl, transpose_B=True, clear_accum=True)
            for i, j in T.serial(M, R):
                hl[i, j] = hl[i, j] * s
            T.gemm(hl, bl, cl, clear_accum=False)
            T.copy(cl, C[0, 0])
    k = cmp_(main, [4])
    x, w, a, b = torch.rand(8, 8), torch.rand(8, 8), torch.rand(4, 8), torch.rand(4, 8)
    ref = x @ w.T + ((x @ a.T) * 2.0) @ b.T
    return "lora_err=%.2e" % float((k(x, w, a, b, 2.0) - ref).abs().max())


def gdn_recurrent():
    """GDN 前向递推（单头单块）：S ← g·S + k⊗(β(v−kᵀS))，o = qᵀS。"""
    DK, DV, T_ = 8, 8, 4

    @T.prim_func
    def main(Q: T.Tensor((T_, DK), "float32"), K: T.Tensor((T_, DK), "float32"), V: T.Tensor((T_, DV), "float32"),
             AL: T.Tensor((T_,), "float32"), BE: T.Tensor((T_,), "float32"), O: T.Tensor((T_, DV), "float32")):
        with T.Kernel(1) as bx:
            s = T.alloc_local((DK, DV), "float32")
            ql = T.alloc_local((T_, DK), "float32")
            kl = T.alloc_local((T_, DK), "float32")
            vl = T.alloc_local((T_, DV), "float32")
            al = T.alloc_local((T_,), "float32")
            be = T.alloc_local((T_,), "float32")
            pred = T.alloc_local((DV,), "float32")
            T.copy(Q[0, 0], ql)
            T.copy(K[0, 0], kl)
            T.copy(V[0, 0], vl)
            T.copy(AL[0], al)
            T.copy(BE[0], be)
            T.fill(s, T.float32(0))
            for t in T.serial(T_):
                for j in T.serial(DV):
                    pred[j] = T.float32(0)
                    for i in T.serial(DK):
                        pred[j] = pred[j] + kl[t, i] * s[i, j]
                for i in T.serial(DK):
                    for j in T.serial(DV):
                        s[i, j] = al[t] * s[i, j] + kl[t, i] * (be[t] * (vl[t, j] - pred[j]))
                for j in T.serial(DV):
                    O[t, j] = T.float32(0)
                    for i in T.serial(DK):
                        O[t, j] = O[t, j] + ql[t, i] * s[i, j]
    k = cmp_(main, [5])
    q, kk, v = torch.rand(T_, DK), torch.rand(T_, DK), torch.rand(T_, DV)
    al, be = torch.rand(T_), torch.rand(T_)
    o = k(q, kk, v, al, be)
    s = torch.zeros(DK, DV)
    ref = torch.zeros(T_, DV)
    for t in range(T_):
        pred = kk[t] @ s
        s = al[t] * s + torch.outer(kk[t], be[t] * (v[t] - pred))
        ref[t] = q[t] @ s
    return "gdn_fwd_err=%.2e" % float((o - ref).abs().max())


CASES = {"ln16": ln_fp16_out, "rope": rope_inplace, "lora": lora_side, "gdn": gdn_recurrent}

for nm in (sys.argv[1:] or sorted(CASES)):
    go(nm, CASES[nm])
