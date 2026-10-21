#!/usr/bin/env python3
"""reconcile/U_oracle.py — 代理 U：letter_readout 数值诊断工装（无卡可闭环的六档尺子）。

【做什么】把 `letter_readout_asc.py` 的两条 ascend 正文（`gather_asc_impl` /
`scatter_add_asc_impl`）与 cpu 正文（数值 oracle）在同一份工装里对齐三件事：
  ① --shadow 影子重放：用纯 python 逐行复刻 ascend 正文的下标算术（块号×块宽+块内号、
     尾块守卫、散取行偏移、逐格读—加—落），与 torch 尺子（rows[ids] / index_add_）对拍。
     这一档不需要 tilelang，只回答一件事：**正文写下的下标算术本身对不对**。
  ② --cpu 双方言 oracle：把正文写成"方言无关生成器"（T 由 ascend_env.dialect(target) 注入，
     只用两方言公共子集 T.serial / T.Kernel / T.ceildiv / 标量算术 / 下标读写），
     用 target=cpu **真编译真发射**，再与 torch 尺子对拍。这一档把"源级语义"从
     昇腾 codegen 里剥离出来：cpu 真跑绿 ⇒ 偏差只能在 ascend 下发射，不在算式。
  ③ --dump 生成码体检：容器内 target=ascend 真编译，把生成码打出来并做关键位探针
     （行偏移是否 `Idx[row]*dim + j`、尾块守卫是否存在、位宽是否 fp32 直通、
     标量读写走的是 dcache 路还是 gm_bypass 路）。这是**无卡条件下唯一能看见
     "ascend 把这句话下射成什么"的凭据**。
  ④ --b-shadow / --b-dump：真机 FAIL 那条 case 的正文（attempts-B 的 scores 投影）同源对照，
     用来回答"判决件到底量了谁"与 fp16 链底噪有多大。
  ⑤ --harden：三形态（A 逐格重复取号 / B 每行一次取号 / C MTE 整行装填）在 target=ascend 的
     生成码计数判决 + cpu 真跑等价性——选形态 B 的落地依据。
  ⑥ --prod：把 sys.modules 里的 ascend 方言名指向 cpu 方言，**真跑仓库里那两条生产正文**，
     与 ② 的镜像正文互为交叉验证（免镜像漂移）。
【怎么做】判据形状刻意覆盖四类边界：重复行号（同行命中多次）、非整除形状（尾块）、
  行宽非 8 倍数（32B 粒度面）、picked=1/0（退化网格），另加 b 档 scores 形态
  （run_numerics.num_readout 的正文）做同源对照——wave2 真机 FAIL 的那条 case 量的是 b 形态，
  与本件共享的是"动态下标散取 + 标量落回"这一条下射路，必须同档比。
【为什么】真机 rel=0.1777896 既非崩溃也非全零型（≈1），是"部分格子错"的真数值偏差；
  无卡时只能靠"源级正确 + 下射可见"两头夹：源级用 ①② 证伪/证实，下射用 ③ 取证。
  被否方案一：直接在卡上试改试跑——违 R20（禁卡上试错）；被否方案二：只跑既有
  host 单测——那些用例全走 cpu 正文与 torch 回退，ascend 正文从未真执行过，测不到这次的真错。

用法：
  host:   cd release && .venv/bin/python ascend/kernels/reconcile/U_oracle.py --shadow
          .venv/bin/python ascend/kernels/reconcile/U_oracle.py --cpu
  容器:   docker exec cann910b-j bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
          export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
          cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 600 \\
            python3 ascend/kernels/reconcile/U_oracle.py --dump'
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../release/ascend/kernels/reconcile
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import numpy as np  # noqa: E402

#: 与 letter_readout_asc.GATHER_ROWS_PER_BLOCK 同源；诊断期允许从命令行覆盖以便对比不同分块
RPB_DEFAULT = 8
REL_TOL = 1e-5           # 纯搬运应逐位同；留 1e-5 与 run_numerics 的 readout 判据同尺


def relerr(a, b):
    """与 run_numerics.relerr 同式：||a-b|| / (||b|| + 1e-9)。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


# --------------------------------------------------------------------------- 形状用例
def cases():
    """(名称, 表行数 total, 行宽 dim, 行号数组) —— 四类边界各占一列。

    白话：点名册里故意有重复的、有只点一行的、有表宽不是 8 的倍数的、有抽的行数
    除不尽块宽的（留尾块），还有册子空的（退化网格）。
    """
    return [
        ("dup(尾块)", 12, 16, [3, 0, 7, 3, 11, 5]),          # 重复行号 + 6<8 尾块
        ("exact(整除)", 16, 64, [0, 3, 3, 15, 7, 7, 1, 9]),   # picked 恰为 RPB 整数倍
        ("odd-dim", 10, 26, [9, 9, 9, 0, 4, 4, 8, 2, 1]),     # 行宽非 32B 整倍 + 三重命中
        ("single", 8, 32, [5]),                               # picked=1 ⇒ 网格=1 块、7 格守卫
        ("wide-tail", 5, 8, [4, 2, 0, 3, 1, 4, 2, 0, 3, 1, 0, 3]),  # 12 行 ⇒ 1.5 块
        ("empty", 6, 16, []),                                 # picked=0 ⇒ 退化网格
    ]


# --------------------------------------------------------------------------- ① 影子重放
def shadow_gather(rows, ids, rpb):
    """逐行复刻 gather_asc_impl 的下标算术（python 版，不经任何编译器）。"""
    picked, total = len(ids), rows.shape[0]
    out = np.zeros((picked, rows.shape[1]), dtype=np.float32)
    for bx in range((picked + rpb - 1) // rpb):              # T.Kernel(ceildiv(picked, rpb))
        for i in range(rpb):                                 # T.serial(rpb)
            row = bx * rpb + i
            if row < picked:                                 # 尾块守卫
                for j in range(rows.shape[1]):               # T.serial(dim)
                    idx = ids[row]
                    out[row, j] = rows[idx[0] if isinstance(idx, np.ndarray) else idx, j] \
                        if 0 <= idx < total else 0.0         # codegen 界守卫：越界读回 0
    return out


def shadow_scatter(gout, ids, total, rpb=None):
    """逐行复刻 scatter_add_asc_impl：单块串行逐格"读—加—落"（出口须预清零）。"""
    grows = np.zeros((total, gout.shape[1]), dtype=np.float32)
    for i in range(len(ids)):
        for j in range(gout.shape[1]):
            k = ids[i]
            if 0 <= k < total:                               # 同 gather：越界落点由守卫兜
                grows[k, j] = grows[k, j] + gout[i, j]
    return grows


def run_shadow(rpb):
    print(f"[shadow] GATHER_ROWS_PER_BLOCK={rpb} 逐行复刻 ascend 正文下标算术 vs torch 尺子")
    ok = True
    rng = np.random.default_rng(11)
    for name, total, dim, id_list in cases():
        ids = np.asarray(id_list, dtype=np.int32)
        rows = rng.standard_normal((total, dim)).astype(np.float32)
        got = shadow_gather(rows, ids, rpb)
        ref = rows[ids] if len(ids) else np.zeros((0, dim), np.float32)
        r = relerr(got, ref)
        exact = np.array_equal(got, ref)
        # 反向：把 got 当 dy 退回原表
        gout = rng.standard_normal((len(ids), dim)).astype(np.float32)
        sgot = shadow_scatter(gout, ids, total)
        sref = np.zeros((total, dim), np.float32)
        for pos, k in enumerate(ids):                        # numpy 版 index_add
            sref[k] += gout[pos]
        r2 = relerr(sgot, sref) if len(ids) else 0.0
        dup = len(set(ids.tolist())) != len(ids)
        flag = "OK " if (r < REL_TOL and r2 < REL_TOL) else "BAD"
        ok &= flag == "OK "
        print(f"  {flag} {name:11s} total={total:3d} dim={dim:3d} picked={len(ids):3d} "
              f"dup={dup!s:5s} gather_rel={r:.3e} scatter_rel={r2:.3e} gather_bitexact={exact}")
    print(f"[shadow] {'SOURCE-SEMANTICS-PASS' if ok else 'SOURCE-SEMANTICS-FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- ② 双方言生成器
def gather_body(T, Rows, Idx, Out, dim: int, rpb: int):
    """方言无关的 gather 正文：与 letter_readout_asc.gather_asc_impl **逐行同式**。

    只用 cpu/ascend 公共子集（T.dynamic/T.prim_func/T.Tensor/T.Kernel/T.ceildiv/
    T.serial/标量下标读写），于是同一份正文能在 target=cpu 上真发射。
    """
    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_impl(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                    Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：与 ascend 正文同式的动态网格 + 尾块守卫 + 逐格散取。"""
        with T.Kernel(T.ceildiv(picked, rpb)) as bx:
            for i in T.serial(rpb):
                row = bx * rpb + i
                if row < picked:
                    for j in T.serial(dim):
                        Out[row, j] = Rows[Idx[row], j]

    return gather_impl


def scatter_body(T, GOut, Idx, GRows, dim: int):
    """方言无关的 scatter_add 正文：与 letter_readout_asc.scatter_add_asc_impl 逐行同式。"""
    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def scatter_impl(GOut: T.Tensor((picked, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                     GRows: T.Tensor((total, dim), "float32")):
        """被追踪的那一层：单块串行逐格读—加—落（目标表已清零）。"""
        with T.Kernel(1) as bx:
            for i in T.serial(picked):
                for j in T.serial(dim):
                    GRows[Idx[i], j] = GRows[Idx[i], j] + GOut[i, j]

    return scatter_impl


def run_cpu(rpb):
    """target=cpu 真编真跑 ascend 正文形态 ⇒ 与 torch 尺子对拍（数值闭环不等卡）。"""
    import torch
    import tilelang
    from ascend.kernels import ascend_env

    T = ascend_env.dialect(ascend_env.TARGET_CPU)
    kwargs = ascend_env.compile_kwargs(ascend_env.TARGET_CPU)
    print(f"[cpu-dialect] 方言={T.__name__} compile_kwargs={kwargs} rpb={rpb}")
    rng = np.random.default_rng(3)
    ok = True
    for name, total, dim, id_list in cases():
        ids = torch.tensor(id_list, dtype=torch.int32)
        rows = torch.from_numpy(rng.standard_normal((total, dim)).astype(np.float32))
        picked = int(ids.numel())
        if picked == 0:
            print(f"  SKIP {name:11s} picked=0 ⇒ tilelang 运行期 stride 校验拒 "
                  f"Static stride mismatch（既有接口事实，见 U_readout_diag.md 附带发现）")
            continue
        out = torch.zeros((picked, dim), dtype=torch.float32)
        try:
            k = tilelang.compile(gather_body(T, rows, ids, out, dim, rpb), out_idx=[], **kwargs)
            k(rows, ids, out)
            ref = rows[ids.long()] if picked else torch.zeros((0, dim))
            r = relerr(out.numpy(), ref.numpy())
            bitexact = bool(torch.equal(out, ref))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {name:11s} gather: {type(exc).__name__}: {str(exc)[:160]}")
            ok = False
            continue
        gout = torch.from_numpy(rng.standard_normal((picked, dim)).astype(np.float32))
        grows = torch.zeros((total, dim), dtype=torch.float32)
        try:
            kb = tilelang.compile(scatter_body(T, gout, ids, grows, dim), out_idx=[], **kwargs)
            kb(gout, ids, grows)
            sref = torch.zeros((total, dim))
            if picked:
                sref.index_add_(0, ids.long(), gout)
            r2 = relerr(grows.numpy(), sref.numpy())
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {name:11s} scatter: {type(exc).__name__}: {str(exc)[:160]}")
            ok = False
            continue
        good = r < REL_TOL and r2 < REL_TOL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:11s} picked={picked:3d} dim={dim:3d} "
              f"gather_rel={r:.3e} bitexact={bitexact} scatter_rel={r2:.3e} "
              f"dup_hit={len(set(id_list)) != len(id_list)}")
    verdict = 'DUAL-DIALECT-ORACLE-PASS' if ok else 'DUAL-DIALECT-ORACLE-FAIL'
    print(f"[cpu-dialect] {verdict}（镜像正文的下标算术在真执行下与 torch 尺子逐位同）")
    return 0 if ok else 1


# --------------------------------------------------------------------------- ③ 生成码体检
def _src_probe(tag, src, expects):
    """把关键位探针逐条打印：每条 = (标签, 判据 lambda, 说明)。"""
    print(f"[dump:{tag}] len={len(src)} main_kernel={'main_kernel' in src}")
    for label, hit, note in expects:
        print(f"    {label:26s} {hit!s:5s}  {note}")


def _show(src, needle, before=6, after=8, maxhits=3):
    """把生成码里含 needle 的上下文逐段打印（人读凭据，进 U_readout_diag.md）。"""
    lines = src.splitlines()
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    print(f"    ── 命中 '{needle}' ×{len(hits)}，展示前 {min(maxhits, len(hits))} 处 ──")
    for i in hits[:maxhits]:
        seg = lines[max(0, i - before):i + after]
        for ln in seg:
            print(f"      | {ln}")
        print("      ·" * 30)


def run_dump(rpb):
    """容器内 target=ascend 编译两条正文并导出/体检生成码。"""
    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    import torch
    import tilelang
    from ascend.kernels import ascend_env, letter_readout_asc

    TA = ascend_env.dialect(ascend_env.TARGET_ASCEND)
    rows = torch.randn(16, 64, dtype=torch.float32)
    ids = torch.tensor([0, 3, 3, 15, 7, 7, 1, 9, 2, 4, 4, 11, 5, 6, 8, 13, 12, 14, 10, 15],
                       dtype=torch.int32)                       # 20 行 ⇒ 2.5 块（尾块激励）
    out = torch.zeros((int(ids.numel()), 64), dtype=torch.float32)
    srcs = {}
    for tag, builder in (("gather", lambda: letter_readout_asc.gather_asc_impl(rows, ids, out, 64)),
                         ("scatter", lambda: letter_readout_asc.scatter_add_asc_impl(rows[:20] if rows.size(0) >= 20 else rows, ids, torch.zeros(16, 64), 64))):
        try:
            kern = tilelang.compile(builder(), target="ascend", out_idx=[])
            src = str(kern.get_kernel_source())
        except Exception as exc:  # noqa: BLE001
            print(f"[dump:{tag}] COMPILE-FAIL {type(exc).__name__}: {str(exc)[:300]}")
            return 1
        srcs[tag] = src
        p = os.path.join("/tmp", f"u_{tag}.asc")
        try:
            with open(p, "w") as f:
                f.write(src)
            print(f"[dump:{tag}] 生成码落 {p}")
        except OSError:
            pass

    _src_probe("gather", srcs["gather"], [
        ("idx*dim 行偏移", ("Idx" in srcs["gather"]), "散取起点是否来自 Idx 的读出"),
        ("界守卫", ("0 <=" in srcs["gather"] or "< 0" in srcs["gather"]), "codegen 是否为动态下标补守卫"),
        ("尾块守卫 row<picked", ("picked" in srcs["gather"]), "动态网格尾块是否真被挡住"),
        ("gm_bypass 标量读", ("bypass" in srcs["gather"]), "纯读表走 dcache 还是 bypass"),
        ("批量搬运件", ("asc_copy_gm2ub_align" in srcs["gather"]), "是否误走 32B 粒度批量路"),
        ("simt/atomic", ("SimtVF" in srcs["gather"] or "asc_atomic" in srcs["gather"]), "910B 无载体形态"),
    ])
    for needle in ("Idx", "picked", "bypass"):
        _show(srcs["gather"], needle)
    _src_probe("scatter", srcs["scatter"], [
        ("读—加—落 RMW", ("+" in srcs["scatter"]), "累加是否为读出+新值"),
        ("gm_bypass", ("bypass" in srcs["scatter"]), "RMW 是否绕 dcache（否则丢账）"),
    ])
    for needle in ("Idx", "bypass"):
        _show(srcs["scatter"], needle)
    print("[dump] U-DUMP-DONE")
    return 0


# --------------------------------------------------------------------------- ④ b 形态同源对照
def b_body(T, BATCH=8, SEQ=32, DIM=512, KK=26, BM=4, HDT="float16", VOCAB=4096):
    """run_numerics.num_readout / attempts-B 的正文（scores 投影）—— 与本件共享散取下射路。

    只作对照用（本波不改它）：wave2 真机 FAIL 的 rel=0.1777896 量的是这份正文，
    其中 `w_ub[t, j] = cast(W[IDS[t], j])` 与本件 `Out[row,j] = Rows[Idx[row],j]`
    是**同一条动态下标散取**，`h_ub[i,j] = HID[bx*BM+i, POS[bx*BM+i], j]` 多一层中间轴动态。
    """
    F32, I32 = "float32", "int32"

    @T.prim_func
    def main(HID: T.Tensor((BATCH, SEQ, DIM), HDT), POS: T.Tensor((BATCH,), I32),
             W: T.Tensor((VOCAB, DIM), HDT), IDS: T.Tensor((KK,), I32),
             S: T.Tensor((BATCH, KK), F32)):
        """b 档 scores 正文（逐字照抄 run_numerics::num_readout.build）。"""
        with T.Kernel(T.ceildiv(BATCH, BM)) as bx:
            h_ub = T.alloc_shared((BM, DIM), F32)
            w_ub = T.alloc_shared((KK, DIM), F32)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    h_ub[i, j] = T.cast(HID[bx * BM + i, POS[bx * BM + i], j], F32)
            for t in T.serial(KK):
                for j in T.serial(DIM):
                    w_ub[t, j] = T.cast(W[IDS[t], j], F32)
            for i in T.serial(BM):
                acc = T.alloc_var(F32)
                for t in T.serial(KK):
                    acc = 0.0
                    for j in T.serial(DIM):
                        acc = acc + h_ub[i, j] * w_ub[t, j]
                    S[bx * BM + i, t] = acc

    return main


def run_b_shadow():
    """b 形态的影子重放：python 复刻其下标算术与累加序，vs f64 尺子 ⇒ 判"源级 vs 下射"。"""
    BATCH, SEQ, DIM, KK, BM, VOCAB = 8, 32, 512, 26, 4, 4096
    rng = np.random.default_rng(7)
    hid = rng.standard_normal((BATCH, SEQ, DIM)).astype(np.float16).astype(np.float32)
    w = rng.standard_normal((VOCAB, DIM)).astype(np.float16).astype(np.float32)
    pos = rng.integers(0, SEQ, size=BATCH).astype(np.int32)
    ids = np.arange(KK, dtype=np.int32)

    # ① 正文算式的 python 复刻（含 BM 分块、逐格散取、fp32 顺序累加）
    s = np.zeros((BATCH, KK), np.float32)
    for bx in range((BATCH + BM - 1) // BM):
        h_ub = np.zeros((BM, DIM), np.float32)
        w_ub = np.zeros((KK, DIM), np.float32)
        for i in range(BM):
            r = bx * BM + i
            for j in range(DIM):
                h_ub[i, j] = np.float32(hid[r, pos[r], j])
            for t in range(KK):
                for j in range(DIM):
                    w_ub[t, j] = np.float32(w[ids[t], j])
        for i in range(BM):
            for t in range(KK):
                acc = np.float32(0.0)
                for j in range(DIM):
                    acc = np.float32(acc + np.float32(h_ub[i, j] * w_ub[t, j]))
                s[bx * BM + i, t] = acc
    ref = (hid[np.arange(BATCH), pos, :].astype(np.float64)
           @ w[ids, :].astype(np.float64).T).astype(np.float32)
    bad = (hid[:, 0, :].astype(np.float64) @ w[ids, :].astype(np.float64).T).astype(np.float32)
    r_self = relerr(s, ref)
    r_bad = relerr(s, bad)
    print(f"[b-shadow] 源级复刻 rel={r_self:.3e}  rel_firstpos={r_bad:.3e} "
          f"bitexact={np.array_equal(s, ref)}")
    print(f"[b-shadow] 参照真机：NUM-readout-FAIL rel=1.777896e-01（wave2）⇒ "
          f"源级偏差量级 vs 真机量级 = {r_self:.2e} vs 1.78e-01")
    rows16 = hid[np.arange(BATCH), pos, :].astype(np.float16)
    w16 = w[ids, :].astype(np.float16)
    prod = (rows16[:, None, :] * w16[None, :, :]).astype(np.float16)
    acc16 = np.zeros((BATCH, KK), np.float16)
    for j in range(DIM):
        acc16 = (acc16 + prod[:, :, j]).astype(np.float16)
    r_fp16 = relerr(acc16.astype(np.float32), ref)
    msg = "[b-shadow] 全 fp16 乘加链底噪 rel=" + format(r_fp16, ".3e") + "（供排除 dtype 链假设）"
    print(msg)
    print(f"[b-shadow] {'B-SOURCE-SEMANTICS-PASS' if r_self < REL_TOL else 'B-SOURCE-SEMANTICS-FAIL'}")
    return 0


def run_b_dump():
    """容器内把 b 形态编到 target=ascend，导出其生成码，与本件 gather 的下射逐段比。"""
    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    import tilelang
    from ascend.kernels import ascend_env

    TA = ascend_env.dialect(ascend_env.TARGET_ASCEND)
    kern = tilelang.compile(b_body(TA), target="ascend", out_idx=-1)
    src = str(kern.get_kernel_source())
    with open("/tmp/u_b.asc", "w") as f:
        f.write(src)
    print(f"[dump:b] 生成码 len={len(src)} 落 /tmp/u_b.asc "
          f"bulk_copy={'asc_copy_gm2ub_align' in src} bounds_guard={'0 <=' in src or '< 0' in src}")
    for needle in ("POS", "IDS", "h_ub", "w_ub"):
        _show(src, needle, before=3, after=5, maxhits=2)
    print("[dump:b] U-B-DUMP-DONE")
    return 0


def main():
    argv = sys.argv[1:]
    mode = next((a.lstrip("-") for a in argv if a.startswith("--") and a != "--"), "shadow")
    rpb = next((int(a.split("=")[1]) for a in argv if a.startswith("--rpb=")), RPB_DEFAULT)
    if mode == "shadow":
        return run_shadow(rpb)
    if mode == "cpu":
        return run_cpu(rpb)
    if mode == "dump":
        return run_dump(rpb)
    if mode == "b-shadow":
        return run_b_shadow()
    if mode == "b-dump":
        return run_b_dump()
    if mode == "harden":
        return run_harden(rpb)
    if mode == "prod":
        return run_prod(rpb)
    print(f"未知模式 --{mode}（可用：shadow / cpu / prod / dump / b-shadow / b-dump / harden，可加 --rpb=N）")
    return 2



# --------------------------------------------------------------------------- ⑤ 形态对照（下射取证）
def variant_scalar(T, Rows, Idx, Out, dim, rpb):
    """形态 A（现行）：逐格 `Rows[Idx[row], j]` —— 界守卫与偏移各自重复载入 Idx。

    与 letter_readout_asc.gather_asc_impl 逐字同式，只作对照基线。
    """
    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_v1(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                  Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：现行形态基线。"""
        with T.Kernel(T.ceildiv(picked, rpb)) as bx:
            for i in T.serial(rpb):
                row = bx * rpb + i
                if row < picked:
                    for j in T.serial(dim):
                        Out[row, j] = Rows[Idx[row], j]

    return gather_v1


def variant_hoist(T, Rows, Idx, Out, dim, rpb):
    """形态 B：行号先落一次标量变量，守卫与偏移都读寄存器（语义逐位不变）。

    动机：形态 A 的生成码里每个出口元素要重复向 GM 取三次同一个行号（守卫两次 + 偏移一次），
    行数×行宽×3 次"数据相关"标量散读是 910B 上唯一没有真机先例的下射形态。
    """
    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_v2(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                  Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：一行一次取号，逐格散取照旧。"""
        with T.Kernel(T.ceildiv(picked, rpb)) as bx:
            k = T.alloc_var("int32")
            for i in T.serial(rpb):
                row = bx * rpb + i
                if row < picked:
                    k = Idx[row]
                    for j in T.serial(dim):
                        Out[row, j] = Rows[k, j]

    return gather_v2


def variant_copy(T, Rows, Idx, Out, dim, rpb):
    """形态 C：整行走 MTE 批量搬运进 UB，出口仍逐格标量落（真机绿件的两条被证明腿）。

    动机：真机六条绿案（addln/rope/attn_sw/gdn-conv/delta/gemm）里，"把 GM 数据搬进核内"
    这一腿要么走 `T.copy`（MTE/DMA，直读物理 GM、不经核侧 dcache），要么其标量读都是
    **仿射下标**；唯有 readout 用的是"载入值当下标"的间接散读。形态 C 把散读换成
    按行动态起点的批量搬 —— 起点仍是 `Idx[row]`（载入值），但数据腿改走 DMA。
    约束：GAP-B 的 `asc_copy_gm2ub_align` 有 32B 粒度静默截断，行字节数须 32 整倍
    （fp32 ⇒ dim%8==0）；不满足时不可用本形态。
    """
    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_v3(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                  Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：一次一行搬进 UB，再逐格原样落出口。"""
        with T.Kernel(T.ceildiv(picked, rpb)) as bx:
            line = T.alloc_shared((dim,), "float32")
            for i in T.serial(rpb):
                row = bx * rpb + i
                if row < picked:
                    T.copy(Rows[Idx[row], 0], line)
                    for j in T.serial(dim):
                        Out[row, j] = line[j]

    return gather_v3


def run_harden(rpb):
    """三形态对照判决：cpu 方言真跑（数值等价性）+ ascend 编译判决 + 生成码载入计数。"""
    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    import torch
    import tilelang
    from ascend.kernels import ascend_env

    forms = (("A 现行标量散读", variant_scalar), ("B 取号收敛标量", variant_hoist),
             ("C MTE 整行装填", variant_copy))
    total, dim, picked = 24, 64, 20
    ids_np = [0, 3, 3, 15, 7, 7, 1, 9, 2, 4, 4, 11, 5, 6, 8, 13, 12, 14, 10, 15]

    print(f"[harden] 形状 rows=({total},{dim}) picked={picked} rpb={rpb} ids 含重复/尾块")
    for tag, builder in forms:
        # ── cpu 方言真跑：与 torch 尺子逐位比（数值等价性凭据）
        TC = ascend_env.dialect(ascend_env.TARGET_CPU)
        rows = torch.arange(total * dim, dtype=torch.float32).reshape(total, dim)
        ids = torch.tensor(ids_np, dtype=torch.int32)
        out = torch.full((picked, dim), float("nan"), dtype=torch.float32)
        try:
            kc = tilelang.compile(builder(TC, rows, ids, out, dim, rpb), out_idx=[],
                                  **ascend_env.compile_kwargs(ascend_env.TARGET_CPU))
            kc(rows, ids, out)
            want = rows[ids.long()]
            ok = bool(torch.equal(out, want))
            rel = relerr(out.numpy(), want.numpy())
            print(f"  {tag:16s} cpu真跑 bitexact={ok!s:5s} rel={rel:.3e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:16s} cpu真跑 ERR {type(exc).__name__}: {str(exc)[:140]}")

        # ── ascend 编译判决 + 生成码计数
        TA = ascend_env.dialect(ascend_env.TARGET_ASCEND)
        out2 = torch.zeros((picked, dim), dtype=torch.float32)
        try:
            ka = tilelang.compile(builder(TA, rows, ids, out2, dim, rpb),
                                  **ascend_env.compile_kwargs(ascend_env.TARGET_ASCEND),
                                  out_idx=[])
            src = str(ka.get_kernel_source())
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:16s} ascend COMPILE-FAIL {type(exc).__name__}: {str(exc)[:200]}")
            continue
        idx_loads = src.count("Idx[")
        print(f"  {tag:16s} ascend PASS len={len(src)} 每行取号文本计数={idx_loads}（循环体计数，j 环未展开） "
              f"MTE批量={'asc_copy_gm2ub' in src} 标量出口={'write_gm_bypass_dcache' in src} "
              f"界守卫={'0 <=' in src}")
    print("[harden] U-HARDEN-DONE")
    return 0


# --------------------------------------------------------------------------- ⑥ 生产正文真跑（免镜像漂移）
def run_prod(rpb):
    """把**仓库里那两条生产 ascend 正文**在 host 以 cpu 方言真编译真发射。

    手法：`gather_asc_impl` / `scatter_add_asc_impl` 的函数体里是 `import tilelang.ascend.language
    as T`。先把 sys.modules 中这个名字指向 cpu 方言模块（并伪装好父包），同一份正文就以
    target="c" 编译并真跑——测的是**生产代码本身**，与 ② 的"镜像正文"互为交叉验证：
    一旦有人改了生产件而镜像没跟上，这一档会立刻分歧。

    口径边界（如实说明，不夸大）：
      - 本档只判**下标算术与累加语义**（方言公共子集部分）；ascend 特有下射（标量 GM 散读、
        界守卫、bypass store）由 ③/⑤ 的生成码体检负责。
      - `picked=0` 的退化形状会被 tilelang 运行期 stride 校验挡在发射之前（既有接口事实，
        与本波改动无关），这里记为 SKIP 而不是 FAIL。
    """
    import sys
    import types

    import torch
    import tilelang
    import tilelang.cpu.language as tcpu
    from ascend.kernels import ascend_env, letter_readout_asc

    if "tilelang.ascend" not in sys.modules:
        parent = types.ModuleType("tilelang.ascend")
        parent.__path__ = []                       # 伪装成包，够 import 机制用
        sys.modules["tilelang.ascend"] = parent
    sys.modules["tilelang.ascend.language"] = tcpu

    kwargs = ascend_env.compile_kwargs(ascend_env.TARGET_CPU)
    rpb_eff = letter_readout_asc.GATHER_ROWS_PER_BLOCK
    print(f"[prod] 生产正文真跑 方言={tcpu.__name__} GATHER_ROWS_PER_BLOCK={rpb_eff}"
          f"（② 镜像档用 rpb={rpb}）")
    rng = np.random.default_rng(5)
    ok = True
    extra = ("dup-heavy", 9, 32, [2, 2, 2, 8, 0, 0, 5, 5, 5, 1, 2, 8])   # 同行最多命中 4 次
    for name, total, dim, id_list in cases() + [extra]:
        ids = torch.tensor(id_list, dtype=torch.int32)
        rows = torch.from_numpy(rng.standard_normal((total, dim)).astype(np.float32))
        picked = int(ids.numel())
        if picked == 0:
            print(f"  SKIP {name:11s} picked=0 ⇒ tilelang 运行期 stride 校验拒 "
                  f"Static stride mismatch（见 U_readout_diag.md 附带发现）")
            continue
        out = torch.full((picked, dim), float("nan"), dtype=torch.float32)
        try:
            prog = letter_readout_asc.gather_asc_impl(rows, ids, out, dim)
            tilelang.compile(prog, out_idx=[], **kwargs)(rows, ids, out)
            ref = rows[ids.long()]
            r = relerr(out.numpy(), ref.numpy())
            bitexact = bool(torch.equal(out, ref))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {name:11s} gather: {type(exc).__name__}: {str(exc)[:150]}")
            ok = False
            continue
        gout = torch.from_numpy(rng.standard_normal((picked, dim)).astype(np.float32))
        grows = torch.zeros((total, dim), dtype=torch.float32)
        try:
            prog2 = letter_readout_asc.scatter_add_asc_impl(gout, ids, grows, dim)
            tilelang.compile(prog2, out_idx=[], **kwargs)(gout, ids, grows)
            sref = torch.zeros((total, dim))
            sref.index_add_(0, ids.long(), gout)
            r2 = relerr(grows.numpy(), sref.numpy())
            sexact = bool(torch.equal(grows, sref))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {name:11s} scatter: {type(exc).__name__}: {str(exc)[:150]}")
            ok = False
            continue
        good = r < REL_TOL and r2 < REL_TOL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:11s} total={total:3d} dim={dim:3d} "
              f"picked={picked:3d} gather_rel={r:.3e} gather_bitexact={bitexact!s:5s} "
              f"scatter_rel={r2:.3e} scatter_bitexact={sexact} "
              f"dup={len(set(id_list)) != len(id_list)}")
    print(f"[prod] {'PROD-BODY-ORACLE-PASS' if ok else 'PROD-BODY-ORACLE-FAIL'}"
          f"（生产件正文自身，非镜像复刻）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
