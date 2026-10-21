#!/usr/bin/env python3
"""L4B_probe_oneshot — **下窗一枪**（设计已定，本件**不执行**）：以「发射前预热/清零」一次复判三缺陷。

三缺陷（K4 统一假说 GDN 族 device-only 缺陷群）：
  · gradcheck-gdn_conv   （窗四 FAIL max_rel=5.538e+04）
  · gradcheck-gdn_delta  （窗四 PASS/假绿 nan；窗三 FAIL nan；V2 分诊 rel=2.21e5）
  · delta_bwd 的 dbeta   （K4：rel=0.80；已证 prefill=zero 转对 ⇒ 锚点）

【三臂】每缺陷各跑一遍：
  A0 baseline —— 现口径单发射（复现当前判词）。
  A1 UB 预清零 —— 判件发射前把该内核用到的 UB 区段显式写 0（哑发射/前置 zeroing；
                  实现见 `_prezero_ub(op)` 钩子，需 M4 提供 true UB 清零入口，否则退化为 A2 语义）。
  A2 预热发射 —— 先跑一次同形哑发射（预热），再发射判件。

【判别规则（写死）】
  A1 转绿                 ⇒ 字面「未初始化 UB 读」→ 统一假说 +1（修法＝UB 预清零）。
  A1 不转、A2 转          ⇒ 非 UB 内容问题，指向「首发射类」（地址补丁/分配/DCache 一致性）。
  A1/A2 均不转            ⇒ 独立根因。

【对 conv 的强预测（可证伪）】`conv_bwd` 唯一 UB 槽 `dw_ub` 已在源码 261 / 生成码 22 显式清零，
  UB 预清零对其「无可修之物」⇒ **预测 A1 不转绿** ⇒ conv 指向独立根因。若实测转绿，则
  B.3「静态审计无判别力」论断被推翻，须回炉重审计。

跑法（卡上，一枪三件×三臂）：
    cd /work && ASCEND_NPU_ARCH=dav-2201 python3 ascend/kernels/reconcile/L4B_probe_oneshot.py
host 只读计划（**无卡不发射**，打印本件将要做什么；不碰设备）：
    cd release && .venv/bin/python ascend/kernels/reconcile/L4B_probe_oneshot.py --plan
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (RELEASE, os.path.join(RELEASE, "ascend")):
    if p not in sys.path:
        sys.path.insert(0, p)

#: 三缺陷（op, 备注）——gdn_conv/gdn_delta 走 gradcheck；dbeta 走 delta_bwd 判件（另件，仅记锚点）
DEFECTS = (
    ("gradcheck-gdn_conv", "窗四 FAIL max_rel=5.538e+04"),
    ("gradcheck-gdn_delta", "窗四 PASS 假绿 nan / V2 rel=2.21e5"),
    ("delta_bwd-dbeta", "K4 rel=0.80（已证 prefill=zero 转对，锚点）"),
)

#: 预测表（写死，供下窗对拍；见 L4B_RESULT.md §B.4）
PREDICT = {
    "delta_bwd-dbeta": {"A1": "green", "note": "锚点（已证）"},
    "gradcheck-gdn_delta": {"A1": "unknown", "note": "A1 绿⇒支持；不绿⇒独立根因"},
    "gradcheck-gdn_conv": {"A1": "red", "note": "强预测：dw_ub 已清零，UB 预清零无可修之物"},
}


def _prezero_ub(op):
    """A1 钩子：把 op 对应内核用到的 UB 区段显式写 0。

    【为何是钩子】UB 基址清零需内核侧入口（M4 写面），本件不臆造实现：
      · conv_bwd：唯一 UB 槽 dw_ub（源码 261 已自带清零）→ 本钩子对其为 no-op；
      · gdn_forward/bwd：`buf_dyn_shmem` 各槽 —— 需 M4 提供「同形哑发射写 0」入口。
    返回 True 表示已置零；False 表示未实现（此时 A1 退化为 A2 语义，须在结论里标注）。
    """
    return False


def _warm_launch(op):
    """A2：同形哑发射预热（用零输入跑一遍被测内核）。返回 True 表示已预热。"""
    return False


def _run_arm(op, arm, target, device, n_dirs):
    """跑一臂，返回 (verdict, rel)。实现桩：下窗在卡上补 `gc.run_one` / `run_delta_card` 调用。"""
    raise NotImplementedError(
        f"下窗实现：op={op} arm={arm} target={target} device={device} n_dirs={n_dirs}")


def _print_plan():
    print("=== L4B-ONESHOT PLAN（设计；本件不执行）===")
    print("三臂：A0 baseline / A1 UB 预清零 / A2 预热发射")
    print(f"{'缺陷':28s} {'现状':34s} {'A1 预测':8s} 备注")
    for op, cur in DEFECTS:
        pr = PREDICT.get(op, {})
        print(f"{op:28s} {cur:34s} {pr.get('A1','?'):8s} {pr.get('note','')}")
    print("\n判别：A1 绿⇒支持统一假说；A1 不绿且 A2 绿⇒首发射类；二者均不绿⇒独立根因。")
    print("判件阈值/rtol/atol/INVALID 语义**一字不改**（沿用 gradcheck_card）。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="L4B 下窗一枪（三缺陷×三臂；设计件，默认不执行）")
    ap.add_argument("--plan", action="store_true", help="host 只读打印计划（无卡不发射）")
    ap.add_argument("--target", default="ascend", choices=("cpu", "ascend"))
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--execute", action="store_true", help="卡上真跑（需下窗补齐 _run_arm 实现）")
    args = ap.parse_args(argv)

    if args.plan or not args.execute:
        _print_plan()
        return 0

    import torch  # noqa: E402
    from ascend.kernels import ascend_env  # noqa: E402
    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    ascend_env.reset()
    target = ascend_env.normalize_target(args.target)
    npu_ok = hasattr(torch, "npu") and bool(getattr(torch.npu, "is_available", lambda: False)())
    device = torch.device("npu") if (target == ascend_env.TARGET_ASCEND and npu_ok) else torch.device("cpu")
    print(f"=== L4B-ONESHOT target={target} device={device.type} dirs={args.dirs} ===")
    for op, _cur in DEFECTS:
        for arm in ("A0", "A1", "A2"):
            if arm == "A1":
                _prezero_ub(op)
            elif arm == "A2":
                _warm_launch(op)
            try:
                verdict, rel = _run_arm(op, arm, target, device, args.dirs)
                print(f"L4B-ONESHOT {op} arm={arm} verdict={verdict} rel={rel}", flush=True)
            except NotImplementedError as exc:
                print(f"L4B-ONESHOT {op} arm={arm} UNIMPLEMENTED ({exc})", flush=True)
    print("L4B-ONESHOT-TABLE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
