#!/usr/bin/env python3
"""attempts/X_carve_hivmc.py — p2-13 P1-1j 路径1：hivmc 内嵌 bitcode 反解。

方法沿用 L 代理已证流程（P1-4b set_l1_2d 位段即由此实证）：
  1) 在 hivmc 二进制里按 `BC\\xc0\\xde` 魔数切块（块界=下一魔数偏移或 EOF），
     逐块喂 llvm-link 试链（能链上的是合法 bitcode 模块）；
  2) bisheng -x ir -S -emit-llvm 反汇编成 .ll 文本；
  3) grep `_mlir_ciface_load_cbuf_to_ca` / `_mlir_ciface_copy_gm_to_cbuf`
     （含 _transpose/_s4 变体）的实现体，读参数打包/位段。

复跑（容器 cann910b-k）：
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    python3 /tmp/X_carve_hivmc.py'   # 产物落 /tmp/xcarve/
"""
import os
import re
import subprocess

HIVMC = "/usr/local/Ascend/cann-8.5.0/tools/bishengir/bin/hivmc"
LINK = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/llvm-link"
BISHENG = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
OUT = "/tmp/xcarve"
os.makedirs(OUT, exist_ok=True)

data = open(HIVMC, "rb").read()
print("hivmc size:", len(data))
starts = [m.start() for m in re.finditer(rb"BC\xc0\xde", data)]
print("BC magic offsets:", starts)

targets = ("load_cbuf_to_ca", "load_cbuf_to_cb", "copy_gm_to_cbuf", "load_gm_to_cbuf")

for i, off in enumerate(starts):
    end = starts[i + 1] if i + 1 < len(starts) else len(data)
    blob = data[off:end]
    bc = "%s/chunk%d.bc" % (OUT, i)
    open(bc, "wb").write(blob)
    linked = "%s/linked%d.bc" % (OUT, i)
    p = subprocess.run([LINK, bc, "-o", linked], capture_output=True, text=True)
    if p.returncode != 0:
        print("chunk%d (%d B): llvm-link FAIL (%s)" % (i, len(blob), p.stderr.strip().splitlines()[:1]))
        continue
    ll = "%s/dump%d.ll" % (OUT, i)
    q = subprocess.run([BISHENG, "-x", "ir", "-S", "-emit-llvm", linked, "-o", ll],
                       capture_output=True, text=True)
    if q.returncode != 0:
        print("chunk%d: dis FAIL %s" % (i, q.stderr.strip().splitlines()[:2]))
        continue
    txt = open(ll, errors="replace").read()
    hits = {t: txt.count(t) for t in targets}
    names = re.findall(r"@[\w$]*?%s[\w$]*" % "|".join(targets), txt)
    print("chunk%d (%d B): link OK, ll=%d B, hits=%s" % (i, len(blob), len(txt), hits))
    uniq = sorted(set(n.lstrip("@") for n in names))
    for u in uniq[:30]:
        print("   FN:", u)
