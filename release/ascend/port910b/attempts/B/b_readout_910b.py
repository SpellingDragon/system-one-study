#!/usr/bin/env python3
"""attempts/B/b_readout_910b.py — readout（末位读出 × 字母行投影 = scores）的 910B tilelang DSL kernel

语义对齐 sys1/decision/readout.py 的 option_scores（第 90-114 行）：
  pos       = lengths - 1                       ← host 侧 last_positions() 已算好，这里作为 int32 输入
  rows[i,:] = last_hidden[i, pos[i], :]         ← 逐行按动态下标取"写完最后一字留下的数字"
  letter_rows[t,:] = head_weight[letter_ids[t],:]
  scores[i,t]     = Σ_j float(rows[i,j]) * float(letter_rows[t,j])   ← .float() 语义 = fp32 累加

**范围声明（需编排者知悉，见 RESULT.md ③）**：本 kernel 只出 `scores`（Readout.scores），
**不含 softmax/to_probs**。原因：910B(dav-2201) vector 面标量数学件没有 `expf`/`exp`/`log`
（probe_math_symbols.sh 实测 MISSING；只有向量件 __cce_scalar::vexp，需在 VF 块里走，
而 T.Parallel 在 Ascend 方言外圈直接被语义检查拒——见 probe_910b_faces.py P9）。
份额（probs）留 host 侧或后续 VF 块改造，不在本次编译判决范围内。

910B 写法要点：
  · 全部读为**标量 GM 直读 + UB 暂存**（codegen 自动补动态下标界守卫，P5 实测），
    不用 T.copy 批量搬运——readout 的搬运粒度是"按 pos/ids 散取行"，本就不是连续块；
    这同时绕开 GAP-B 搬运件的 32B 粒度约束（compat_gap_B.md 的约束段：K=26 个 int32=104B
    这类非 32 倍数的小搬运若走 T.copy 会被 /32 截断）。
  · Ascend 的 T.Kernel 无 threads= 形参（kernel.py:139）→ 按 batch 分块 bx，块内 T.serial；
  · 累加器 acc 每行复位，逐 (i,t) 点积后标量写 S；无 T.Parallel、无 vector-valued store。

用法（宿主机）：bash attempts/B/run_b.sh b_readout_910b.py 16
可调：R_BATCH R_SEQ R_DIM R_K R_BM R_HDT（hidden/weight 位宽，默认 float16）
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang
import tilelang.ascend.language as T

BATCH = int(os.environ.get("R_BATCH", "8"))
SEQ = int(os.environ.get("R_SEQ", "32"))
DIM = int(os.environ.get("R_DIM", "512"))
K = int(os.environ.get("R_K", "26"))          # 候选字母数（A..Z = 26）
BM = int(os.environ.get("R_BM", "4"))         # 每块行数
HDT = os.environ.get("R_HDT", "float16")      # hidden / head_weight 位宽（bf16 面见 RESULT.md）

F32, I32 = "float32", "int32"
#: UB 占用（字节）= BM*DIM*4（取出的末位行）+ K*DIM*4（字母行 fp32 域）
UB_BUDGET = BM * DIM * 4 + K * DIM * 4


def build():
    @T.prim_func
    def main(HID: T.Tensor((BATCH, SEQ, DIM), HDT), POS: T.Tensor((BATCH,), I32),
             W: T.Tensor((4096, DIM), HDT), IDS: T.Tensor((K,), I32),
             S: T.Tensor((BATCH, K), F32)):
        with T.Kernel(T.ceildiv(BATCH, BM)) as bx:
            h_ub = T.alloc_shared((BM, DIM), F32)   # 本块每行的"末位数字"，已升 fp32
            w_ub = T.alloc_shared((K, DIM), F32)    # k 个字母行，已升 fp32
            # 1) 逐行按 pos 取末位（动态下标散取；界守卫由 codegen 生成）
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    h_ub[i, j] = T.cast(HID[bx * BM + i, POS[bx * BM + i], j], F32)
            # 2) 按 letter_ids 取字母行
            for t in T.serial(K):
                for j in T.serial(DIM):
                    w_ub[t, j] = T.cast(W[IDS[t], j], F32)
            # 3) 点积出 (BM, K) 分数，标量落 GM
            for i in T.serial(BM):
                acc = T.alloc_var(F32)
                for t in T.serial(K):
                    acc = 0.0
                    for j in T.serial(DIM):
                        acc = acc + h_ub[i, j] * w_ub[t, j]
                    S[bx * BM + i, t] = acc

    return main


def main_():
    print(f"B-READOUT-BEGIN [batch={BATCH} seq={SEQ} dim={DIM} k={K} bm={BM} "
          f"hid/weight={HDT} scores=fp32 UB~{UB_BUDGET // 1024}KB]")
    try:
        k = tilelang.compile(build(), target="ascend", out_idx=-1)
    except Exception as e:  # noqa: BLE001
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        errs = [ln for ln in lines if ": error:" in ln or "fatal error" in ln]
        print("B-READOUT-COMPILE-FAIL")
        for ln in (errs[:2] or lines[-2:]):
            print(f"  ! {ln[:260]}")
        return 1
    src = k.get_kernel_source()
    print(f"B-READOUT-GEN len={len(src)} main_kernel={'main_kernel' in src} "
          f"uses_bulk_copy={'asc_copy_gm2ub_align' in src} "
          f"bounds_guard={'0 <=' in src or '< 0' in src} "
          f"expf={'expf' in src} rsqrtf={'rsqrtf' in src}")
    if "main_kernel" not in src:
        print("B-READOUT-COMPILE-FAIL")
        print("  ! 生成码里找不到 main_kernel（发射形态异常）")
        return 1
    print("B-READOUT-COMPILE-PASS")
    return 0


sys.exit(main_())
