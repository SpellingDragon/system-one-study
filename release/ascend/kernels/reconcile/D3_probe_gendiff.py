#!/usr/bin/env python3
"""D3_probe_gendiff.py — P1-1z / D3：GDN 反向 **codegen 差分取证**（host 侧可控实验，不占卡）。

一条命令复现三份生成码，并对"dbeta 相关"部分做**结构**差分（不下因果）：

  cpu/pure    host `--target cpu --bwd` 真跑腿**实际执行**的那份（bug 未复现的那条腿）
  ascend/pure 设备方言 + pure 档（设备可编、非本窗真跑档）
  ascend/ub   设备真跑腿**实际执行**的那份 —— 本窗 C 段判决件 `NUM-delta_bwd FAIL first=dbeta`

机械核查四项（只看结构，不解析语义）：
  C1 **DMA 计数**：`asc_copy_gm2ub`/`copy_gm_to_ubuf`/`copy_ubuf_to_gm`/`DataCopy` 出现次数
     ⇒ 判"该件是否与 S3/E 段的『多突发跨步装填』同面"（0 = 不同面）。
  C2 **UB 槽位分区**：解析 `buf_dyn_shmem` 的常量基址，验证 8 个 `alloc_shared` 槽是否互不重叠。
  C3 **先写后读**：每个槽位首次出现是"写"还是"读"（`[...] = ` vs `= ... [`），防 stale 落点。
  C4 **dbeta 归约原文**：含 `db_acc` 的行（带行号）。

用法：
  python3 D3_probe_gendiff.py                 # 复现三份（需要 docker cann910b-h）
  python3 D3_probe_gendiff.py --no-docker \
      --asc-ub /path/gdnb_ub.asc --asc-pure /path/gdnb_pure.asc --cpu /path/gdnb_cpu.cu
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # .../release/ascend/kernels/reconcile
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../release
sys.path.insert(0, RELEASE)
RUNNER = os.path.join(HERE, "..", "..", "port910b", "run_delta_card.py")   # .../release/ascend/port910b/…
RUNNER = os.path.normpath(RUNNER)
HOST_PY = os.path.join(RELEASE, ".venv", "bin", "python")
CONTAINER = "cann910b-h"
DMA_RE = re.compile(r"asc_copy_gm2ub|copy_gm_to_ubuf|copy_ubuf_to_gm|copy_gm_to_ubuf_align"
                    r"|DataCopy|asc_copy_ub2gm|copy_ubuf_to_gm_align")
# 8 个 alloc_shared 槽按 gdn_asc.gdn_bwd_asc_impl 的 alloc 次序（dk=dv=8 ⇒ lam 64 / 其余各 8）
SLOTS = [("lam_ub", 0, 64), ("dec_ub", 64, 8), ("u_scr", 72, 8), ("dp_scr", 80, 8),
         ("k_ub", 88, 8), ("q_ub", 96, 8), ("v_scr", 104, 8), ("do_scr", 112, 8)]


def _read(p):
    with open(p, "r", errors="replace") as f:
        return f.read()


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _docker_exec(script):
    return subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", script],
                          capture_output=True, text=True)


def gen_ascend(variant, workdir):
    """在容器里真编 ascend 方言 bwd 件（--no-run）并把 gdnb 的 device_kernel.asc 取回本地。"""
    vflag = "" if variant == "ub" else f"--variant {variant}"
    script = (
        "set -u; source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1; "
        "export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; "
        f"rm -rf {workdir} && mkdir -p {workdir}; export TILELANG_CACHE_DIR={workdir}; "
        f"cd /work/ascend/port910b && python3 run_delta_card.py --target ascend --bwd --no-run {vflag} "
        "2>&1 | grep -E 'NUM-delta_bwd|ITEM ' ; "
        "for f in $(find " + workdir + " -name device_kernel.asc); do "
        "  grep -q 'void gdn_bwd_impl_kernel(' $f && echo GDNB-ASC=$f; done"
    )
    r = _docker_exec(script)
    sys.stdout.write(r.stdout)
    m = re.search(r"GDNB-ASC=(\S+)", r.stdout)
    if not m:
        raise SystemExit(f"ascend/{variant}: 未取到 gdnb device_kernel.asc\n{r.stdout}\n{r.stderr}")
    cpath = m.group(1)
    txt = _docker_exec(f"cat {cpath}").stdout
    local = os.path.join(tempfile.gettempdir(), f"d3_gdnb_ascend_{variant}.asc")
    open(local, "w").write(txt)
    return local, txt


def gen_cpu():
    """host 真编 cpu 方言 bwd 件（= 本窗 host 复验腿实际执行的那份 pure）。"""
    cache = tempfile.mkdtemp(prefix="d3_cpu_")
    env = dict(os.environ, TILELANG_CACHE_DIR=cache)
    r = subprocess.run([HOST_PY, RUNNER, "--target", "cpu", "--bwd"],
                       capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
    for ln in r.stdout.splitlines():
        if ln.startswith("ITEM ") or ln.startswith("NUM-delta_bwd"):
            print(ln)
    hit = None
    for root, _d, files in os.walk(cache):
        for fn in files:
            if fn in ("device_kernel.cu", "device_kernel.c"):
                p = os.path.join(root, fn)
                if "gdn_bwd_impl_kernel" in _read(p):
                    hit = p
    if not hit:
        raise SystemExit("cpu: 未取到 gdnb device_kernel.cu")
    return hit, _read(hit)


def analyze(tag, path, txt):
    """对一份生成码打结构核查。返回 dict。"""
    lines = txt.splitlines()
    dma = [i + 1 for i, ln in enumerate(lines) if DMA_RE.search(ln)]
    dbeta = [i + 1 for i, ln in enumerate(lines) if "db_acc" in ln]
    # C2/C3：对 8 个槽位基址，找"首次作为左值写"与"首次被读"的行
    slot_report = []
    for name, base, size in SLOTS:
        # 只认 UB 访问（ascend）或 Lam（cpu 把 lam 放 GM）
        w = rd = None
        for i, ln in enumerate(lines):
            if tag.startswith("ascend") and "buf_dyn_shmem" not in ln:
                continue
            # 该行是否触及这个槽位
            if not re.search(r"(buf_dyn_shmem\)\[|Lam\[|\bLam\b)", ln):
                continue
            if not _touches(ln, base, size, tag):
                continue
            is_write = re.search(r"buf_dyn_shmem\)\[[^\]]*\]\s*=|Lam\[[^\]]*\]\s*=", ln) is not None
            if is_write and w is None:
                w = i + 1
            if not is_write and rd is None:
                rd = i + 1
        slot_report.append((name, base, size, w, rd))
    return {"tag": tag, "path": path, "md5": _md5(path), "nlines": len(lines),
            "dma": dma, "dbeta": dbeta, "slots": slot_report}


def _touches(line, base, size, tag):
    """扫该行所有下标表达式里的整型常量，判是否落在 [base,base+size)。

    codegen 的下标形态有三类，全部要抓（旧版正则只认 `[const]` 一类，对
    `(j_2 + 104)` / `((j * 8) + i)` 漏抓 ⇒ 会误报"首用是读"）：
      1. `[const]`                    常量直下标
      2. `[(var + const)]`            base+offset 型（dec/k/q/v/do/u/dp 各槽）
      3. `[((var * 8) + var)]`        lam 块（dv×dk 展平；常量 8 落在 [0,64) ⇒ 命中 lam）
    工程上只剥 `(int64_t)/(int32_t)` 强转后取全部数字常量，命中即算触及。
    """
    subs = (re.findall(r"buf_dyn_shmem\)\[([^\]]*)\]", line) if tag.startswith("ascend")
            else re.findall(r"Lam\[([^\]]*)\]", line))
    for s in subs:
        s = s.replace("(int64_t)", "").replace("(int32_t)", "")
        for c in (int(x) for x in re.findall(r"\d+", s)):
            if base <= c < base + size:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-docker", action="store_true", help="只用 --asc-ub/--asc-pure/--cpu 指定的文件")
    ap.add_argument("--asc-ub", default=None)
    ap.add_argument("--asc-pure", default=None)
    ap.add_argument("--cpu", default=None)
    a = ap.parse_args()

    print("=" * 78)
    print("D3-CODEGEN-GEN 复现三份生成码（compat ca023c2e；不占卡）")
    print("=" * 78)
    if a.no_docker:
        ub = (a.asc_ub, _read(a.asc_ub))
        pu = (a.asc_pure, _read(a.asc_pure))
        cp = (a.cpu, _read(a.cpu))
    else:
        ub = gen_ascend("ub", "/dev/shm/d3_gd_ub")
        pu = gen_ascend("pure", "/dev/shm/d3_gd_pure")
        cp = gen_cpu()

    rows = [analyze("ascend/ub", *ub), analyze("ascend/pure", *pu), analyze("cpu/pure", *cp)]
    print()
    print("=" * 78)
    print("D3-CODEGEN-TABLE 结构差分（C1 DMA / C4 dbeta 归约行号）")
    print("=" * 78)
    for r in rows:
        print(f"[{r['tag']:12s}] md5={r['md5']} lines={r['nlines']} "
              f"C1-DMA={len(r['dma'])}{r['dma'] if r['dma'] else ''} "
              f"C4-dbeta行={r['dbeta']}")
    print()
    print("C1 判定：三份均 0 条 DMA ⇒ **该件不在 S3/E 段『多突发跨步装填』面上**（无 blockCount 可谈）")
    print()
    print("C2/C3 UB 槽位（基址/大小/首写行/首读行；pure 档无 alloc_shared⇒无此槽，cpu 档 lam 在 GM）：")
    for r in rows:
        print(f"  [{r['tag']}]")
        is_asc = r["tag"].startswith("ascend")
        for name, base, size, w, rd in r["slots"]:
            if not is_asc and name != "lam_ub":
                print(f"    {name:7s} (- 该件无 UB，此槽不适用)")
                continue
            if w is None and rd is None:
                print(f"    {name:7s} (- 该件无 alloc_shared，无此槽)")
                continue
            flag = "" if (w is not None and (rd is None or w <= rd)) else "  <== 首用是读!(疑似 stale 落点)"
            print(f"    {name:7s} base={base:3d} size={size:3d} firstW={w} firstR={rd}{flag}")
    print()
    print("C4 dbeta 归约原文：")
    for r in rows:
        for i, ln in enumerate(_read(r["path"]).splitlines()):
            if "db_acc" in ln:
                print(f"  [{r['tag']:12s}:{i+1}] {ln.strip()}")


if __name__ == "__main__":
    main()
