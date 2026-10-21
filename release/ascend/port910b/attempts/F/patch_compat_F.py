#!/usr/bin/env python3
"""attempts/F/patch_compat_F.py — 把 compat_patch_F.h（GAP-F: 软件 logf）**幂等追加**到
容器 pip 运行副本 port910b_compat.h 文件尾。**绝不写主仓真源**（硬边界）。

用法：
  python3 patch_compat_F.py            # 追加（已存在则跳过）
  python3 patch_compat_F.py --restore  # 从 /tmp/port910b_compat.h.origF 还原快照
  python3 patch_compat_F.py --status   # 只看状态（行数/锚在不在）

前置：首次运行会把未打补丁的快照存成 /tmp/port910b_compat.h.origF（还原基线）。
     本脚本**不做** `cp /tilelang/...`（编排者正在改 trunk §12，容器快照是隔离基线）。
"""
import os
import sys

PIP = ("/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/"
       "port910b_compat.h")
BAK = "/tmp/port910b_compat.h.origF"
ANCHOR = "TL_PORT910B_COMPAT_GAP_F_H"
HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "compat_patch_F.h")


def lines(p):
    with open(p, "r", errors="strict") as f:
        return f.read().count("\n")


def status(tag):
    has = ANCHOR in open(PIP).read()
    print(f"F-PATCH-STATUS {tag}: lines={lines(PIP)} GAP_F_present={has} bak_exists={os.path.exists(BAK)}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--status":
        status("query")
        return 0
    if mode == "--restore":
        if not os.path.exists(BAK):
            print("F-PATCH-RESTORE-SKIP (无备份，说明还没打过补丁)")
            return 0
        with open(BAK) as f:
            data = f.read()
        with open(PIP, "w") as f:
            f.write(data)
        print(f"F-PATCH-RESTORED -> {PIP} ({lines(PIP)} lines)")
        status("after-restore")
        return 0

    if not os.path.exists(BAK):                      # 首次：留还原基线
        with open(PIP) as f:
            cur = f.read()
        with open(BAK, "w") as f:
            f.write(cur)
        print(f"F-PATCH-BAK {BAK} ({cur.count(chr(10))} lines)")

    with open(PIP) as f:
        cur = f.read()
    if ANCHOR in cur:
        print("F-PATCH-ALREADY-PRESENT (幂等跳过)")
        status("after-skip")
        return 0
    with open(PATCH) as f:
        block = f.read()
    before = cur.count("\n")
    with open(PIP, "a") as f:
        f.write("\n" + block if not cur.endswith("\n") else block)
    print(f"F-PATCH-APPENDED {PIP} ({before} -> {lines(PIP)} lines)")
    status("after-apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
