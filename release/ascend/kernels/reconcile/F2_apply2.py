#!/usr/bin/env python3
"""reconcile/F2_apply2.py — F2 第二记定点补丁：fp16 档 Adam 改走 fp32 master weights。

起因（一手复现，见 F2_RESULT ②）：fp16 leaf 直接喂 torch.optim.Adam 时，
`eps=1e-8` 舍入成 0（fp16 装不下）、v_hat 在 |grad|≈7e-6 一档下溢，分母出 0 ⇒
step 后参数 ±inf/nan（train_step fp16 档 loss1=nan 的根源）。这不是接线 bug，是
"梯度累加/优化器态走 fp32"这条位宽纪律在优化器侧的落点：混精训练的生产做法就是
fp32 master weights。本补丁只在 dtype!=fp32 分支引入 master；fp32 档语句逐字不变。
锚点数≠期望则整批不动。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # release/
TS = ROOT / "ascend" / "train_step.py"

OLD = """        opt = torch.optim.Adam(list(p.values()), lr=lr)
        opt.step()
"""
NEW = """        # 优化器：fp32 档逐字维持原样（params 即 leaves，Adam 直接在 fp32 面走）；
        # fp16 档按生产混精口径把**优化器状态与更新放在 fp32 master 副本**上——一手
        # 复现：fp16 leaf 直喂 Adam 时 eps=1e-8 舍入成 0、v_hat 在 |grad|≈7e-6 一档
        # 下溢 ⇒ step 出 ±inf/nan（F2 host 首跑 loss1=nan 的根源）。梯度先把 fp16
        # 原件升宽交给 master（累加面在 fp32），step 完再降回 leaves（出口按输入位宽
        # 交货），与内核侧"C2 §2 归约不降位"是同一纪律的两半。
        params = list(p.values())
        if dtype == torch.float32:
            opt = torch.optim.Adam(params, lr=lr)
            opt.step()
        else:
            master = [t.detach().float().requires_grad_(True) for t in params]
            for m, t in zip(master, params):
                m.grad = None if t.grad is None else t.grad.detach().float()
            torch.optim.Adam(master, lr=lr).step()
            with torch.no_grad():
                for m, t in zip(master, params):
                    t.copy_(m.detach().to(t.dtype))
"""


def main() -> int:
    text = TS.read_text(encoding="utf-8")
    n = text.count(OLD)
    if n != 1:
        print(f"锚点异常：命中 {n}（期望 1），未做任何改动")
        return 2
    TS.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {TS.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
