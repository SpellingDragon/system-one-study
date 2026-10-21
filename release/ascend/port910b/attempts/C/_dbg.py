import numpy as np
f2u = lambda v: np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
u2f = lambda u: np.frombuffer(np.uint32(u).tobytes(), dtype=np.float32)[0]


def em(v):
    kLog2e = np.float32(1.44269504088896340736)
    kLn2 = np.float32(0.69314718055994530942)
    t = np.float32(v * kLog2e)
    n = int(t + 0.5) if t >= 0 else int(t - 0.5)
    r = np.float32(v - np.float32(n) * kLn2)
    p = np.float32(1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666 +
                r * (0.041666666666666664 + r * (0.008333333333333333 +
                r * 0.001388888888888889))))))
    if p < 1.0:
        p = np.float32(p + p)
        n -= 1
    e = n + 127
    return float(v), float(p), n, e, float(u2f(f2u(p) + np.uint32(e << 23))), float(np.exp(v))


for v in [0.0, -1.0, 1.0, -50.0, 50.0, -0.09, 0.3, -20.0, 20.0]:
    print(em(np.float32(v)))

# torch conv cross-check
try:
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    C, L, K = 2, 5, 4
    x = rng.standard_normal((C, L)).astype(np.float32)
    w = rng.standard_normal((C, K)).astype(np.float32)
    b = rng.standard_normal((C,)).astype(np.float32)
    xt = torch.from_numpy(x).unsqueeze(0)
    wt = torch.from_numpy(w).unsqueeze(1)
    y = F.conv1d(F.pad(xt, (K - 1, 0)), wt, torch.from_numpy(b), groups=C)[0].numpy()
    pad = K - 1
    xp = np.pad(x, ((0, 0), (pad, 0)))
    ref = np.zeros((C, L), np.float32)
    for c in range(C):
        for t in range(L):
            ref[c, t] = (w[c] * xp[c, t:t + K]).sum() + b[c]
    print("torch conv:", y[:, 0], "\nnumpy  conv:", ref[:, 0], "\nfull diff:", np.abs(y - ref).max())
except ImportError:
    print("no torch on this host")
