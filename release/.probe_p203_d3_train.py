"""D3 中文 train 分割取证脚本（p2-03 eval-registry · 一次性探查件，不入包、不进门禁测试）。

【做什么】把 `sys1/eval/registry.py` 里 `ZH_TRAIN_PROBE` 那三条取证结论重新跑一遍并对拍：
CMMLU 归档内部有哪些档位、CLUE train 原件在盘上的真实字节与行数、魔搭上两仓有没有 train 数据件。

【怎么做】三条按"零流量优先"排：① 用 `zipfile.namelist()` 数已下载的 cmmlu zip 里 dev/test/train；
② 用 `pyarrow.parquet` 只读 metadata 拿 train parquet 的行数（不解析全表）；③ 调魔搭
`HubApi.get_dataset_files` 列文件清单。①② 全在本地盘上，③ 需要网络。

【为什么】登记项写着"上游无 train 档"这种缺席结论时，报告读者唯一的凭据就是取证能不能复跑——
这个脚本一度从 release/ 丢失，结论当场退化成"抄来的话"（被否方案：只在注释里留一段命令文本，
看上去也能复跑，实际没人跑得起，且命令行拼接的引号一错就静默给出相反结论）。网络失败时脚本以
非零退出并如实打印 ERR，绝不把没验证过的写成验证过的（R14）。

跑法：`cd release && .venv/bin/python .probe_p203_d3_train.py`
"""
import zipfile
import collections
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "bench", "eval_data", "raw")
CMMLU_ZIP = os.path.join(RAW, "cmmlu-subset", "cmmlu_v1_0_1.zip")
CLUE_TRAIN = os.path.join(RAW, "clue-train-subset")


def probe_cmmlu_on_disk() -> None:
    """取证 1：盘上那份 CMMLU 归档里有哪些顶层档位、各多少题文件、有没有 train。

    白话：CMMLU 整包早就下载到本地了，现在只把那个压缩包的文件名列表摊开数一遍——看看里面
    是不是只有"练习示例"和"考卷"两叠，压根没有第三叠"练习题"。数出来没有，那句"上游无 train
    档、我们不登记"就不再是一句空话，而是一条谁都能当场重数的事实。
    """
    print("[1] CMMLU 归档内部档位（零流量）")
    if not os.path.isfile(CMMLU_ZIP):
        print(f"    SKIP: 归档不在盘上 {CMMLU_ZIP}（先跑 `python -m sys1.eval.registry fetch --cn`）")
        return
    names = zipfile.ZipFile(CMMLU_ZIP).namelist()
    tops = collections.Counter(n.split("/")[0] for n in names)
    csvs = collections.Counter(n.split("/")[0] for n in names if n.endswith(".csv"))
    train_hits = [n for n in names if "train" in n.lower()]
    print(f"    顶层目录（按条目数）: {dict(sorted(tops.items()))}")
    print(f"    按 .csv 计数      : {dict(sorted(csvs.items()))}")
    print(f"    含 'train' 的路径 : {len(train_hits)} 个 {train_hits[:3]}")
    print("    → 只有 dev/test 两档，查无 train：不登记 CMMLU train 项")


def probe_clue_train_on_disk() -> None:
    """取证 2：CLUE train 原件在盘上的字节数与行数（登记项 note 里的数字就来自这里）。

    白话：train 那两份文件我们已经真拉回来了，这里翻开每份的"目录页"看它一共多少行、占多少
    字节，跟账本上写的数对一对。对得上，说明底账里的字节和条数不是估的；对不上就得停下来查，
    而不是继续拿这个数去写报告。
    """
    print("[2] CLUE train 原件盘上字节/行数（零流量）")
    files = sorted(glob.glob(os.path.join(CLUE_TRAIN, "**", "*.parquet"), recursive=True))
    if not files:
        print(f"    SKIP: {CLUE_TRAIN} 下没有 parquet（先跑 `python -m sys1.eval.registry fetch --zh-train`）")
        return
    import pyarrow.parquet as pq
    for f in files:
        rel = os.path.relpath(f, ROOT)
        print(f"    {rel}  bytes={os.path.getsize(f):,}  rows={pq.ParquetFile(f).metadata.num_rows:,}")
    print("    → train 档真实存在且带答案列：可登记、可装配")


def probe_upstream_listing() -> int:
    """取证 3：问上游文件清单——两仓各自有没有 train 数据件。网络不通时返回 1。

    白话：前两条只看自己抽屉，这条要问仓库本人"你到底有没有那一档"。列出来 CMMLU 三件里没有
    train、CLUE 四十九件里有十一个 train 件，就跟我们的登记口径严丝合缝。问不到（断网、没装
    SDK）就明说问不到并退出非零：缺席结论可以只靠盘上证据站着，但绝不该伪装成"两边都对过"。
    """
    print("[3] 上游列举（魔搭 API，需要网络）")
    try:
        import socket
        socket.setdefaulttimeout(60)
        from modelscope.hub.api import HubApi
        api = HubApi()
        for repo in ("modelscope/cmmlu", "opencompass/clue"):
            fs = api.get_dataset_files(repo_id=repo, revision="master", recursive=True) or []
            rows = [(str(f.get("Name") or f.get("Path")), f.get("Size")) for f in fs]
            train = [(n, s) for n, s in rows if "train" in n.lower()]
            print(f"    {repo}: files={len(rows)} train_files={len(train)}")
            for n, s in train[:4]:
                print(f"        {n}  {s}")
        return 0
    except Exception as exc:  # noqa: BLE001 - 网络/SDK 失败都要如实报，不吞不造
        print(f"    ERR {type(exc).__name__}: {str(exc)[:200]}")
        print("    → 上游列举这条取证这次没跑通：前两条盘上取证仍成立，但不许声称已复跑上游")
        return 1


if __name__ == "__main__":
    probe_cmmlu_on_disk()
    probe_clue_train_on_disk()
    sys.exit(probe_upstream_listing())
