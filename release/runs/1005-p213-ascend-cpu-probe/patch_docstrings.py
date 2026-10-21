"""给本域所有"被 tilelang 追踪的 prim_func"补上符合注释门（R2）的 docstring。

背景：`scratch/tools/check_comments.py` 的 R2 对所有**非下划线开头**、函数体 >4 行的函数都要求
一段"白话:"（≥30 字、至多引入 1 个黑话词）。本域的方言正文采用的是"外层 xxx_cpu_impl/xxx_asc_impl
+ 内层 @T.prim_func def xxx_impl"的写法（P1 用的是单层 lazy 写法），内层这些 `xxx_impl` 因此同样
受 R2 约束。已在 rope 上实测：给嵌套 prim_func 加 docstring **不影响追踪与数值**（加完对拍仍
fwd 1.19e-07 / roundtrip 0.0 / nc 2 / blk {}）。

脚本按 AST 定位"缺 docstring 的那个函数节点"，在其第一条语句前插入，缩进沿用原语句；同一文件里
同名函数按行号升序取文案（先 CPU 件、后昇腾件）。
"""
import ast
from pathlib import Path

CPU_TAG = "（CPU(c) 件）"
ASC_TAG = "（昇腾件）"

TEXTS = {
    "gemm_impl": [
        """被追踪的那一层：形状在这里落定，只有行数 M 是活的。

        白话""" + CPU_TAG + """：把一大片乘加切成小方块，每个小方块自己算自己那格，
        行数是多少都不影响这套切法；算完先在公共小格子里加上加成、把负数压成零，再整块写回大表格。
        """,
        """被追踪的那一层：块数与流水是编译期常量，行数 M 走动态维。

        白话""" + ASC_TAG + """：同一块乘加切成小方块摊给多个工人，每块搬一次、算一次、
        写一次；谁手里多出来的行不够一整块，就只搬有效的那几行，剩下的格子根本不会被碰。
        """,
    ],
    "dw_impl": [
        """被追踪的那一层：权重表 (N,K) 是常量，样本行数 M 是活的累加轴。

        白话""" + CPU_TAG + """：把每条样本的头信号竖起来，和当时的输入两两对上记总账；
        行数多几条少几条只是"记的次数"不同，表的形状从头到尾没变，所以换批不用重新开模具。
        """,
        """被追踪的那一层：列方向的账按块摊开，行方向串行累加。

        白话""" + ASC_TAG + """：一张大账本按列切成几页，每页自己把这一批的数累上去；
        页数与每页多宽都是定死的，来多少行就累多少轮，不会把别页的数字串了行。
        """,
    ],
    "rope_impl": [
        """被追踪的那一层：头数、每头宽度、转几摞、正转还是反转都是常量，只有条数是活的。

        白话""" + ASC_TAG + """：多个工人各领一段记录，同一条转法照着两张小抄做——前半
        减去后半乘纵向抄，后半加上前半乘横向抄，算完原地放回；第三摞连碰都不碰。
        """,
    ],
    "add_ln_impl": [
        """被追踪的那一层：特征轴宽度是常量，行数走动态维。

        白话""" + CPU_TAG + """：先把两摞纸逐位合成一摞并原样留一份，再算出这一行的平均
        水平和散开程度，用这两把尺子把每格挪到位，最后乘上倍率表、加上小抄表；行数不够一整块
        时，多出来的格子直接跳过，不会拿零去污染统计。
        """,
        """被追踪的那一层：一行一格的统计量放进小格表，搬运按有效行数切片。

        白话""" + ASC_TAG + """：合成、求平均、求散开、挪位、乘倍率加小抄，这一串都在工人
        自己手边的小格子里做完；每轮只搬"真的存在"的那几行，末尾不够一整块的部分根本不进机器。
        """,
    ],
    "ln_bwd_impl": [
        """被追踪的那一层：与正向同一套形状约定，出口多两列按列汇总的账。

        白话""" + CPU_TAG + """：已知"改完的格子该挪多少"，先按行的平均与散开把账倒推回
        合成那一摞的每格，再顺带把这一批对倍率表和小抄表的总影响各累成一列；行数不够整块时
        同样只碰有效行，汇总列的初值在块顶就清好。
        """,
        """被追踪的那一层：每格回推与按列汇总同块完成，汇总走原子累加。

        白话""" + ASC_TAG + """：工人各自把手里这一块的账倒推算完，再把该进总账的两笔
        一笔笔添到公共列上；添的动作是可以并发的，同一列多块来添也不会互相盖掉。
        """,
    ],
    "attn_impl": [
        """被追踪的那一层：头数与每头宽度是常量，序列长度走动态维，窗口写死在比较里。

        白话""" + CPU_TAG + """：每个位置只回头看有限的一段，一块一块地把能看见的内容
        按分数合起来；看不见的（后面的、太远前面的）一律不掺进来，连权重都不给，
        所以"份额总和"始终守着一条线，不会因为整块都被挡着就凭空多出东西。
        """,
        """被追踪的那一层：查询块摊给多个工人，键值沿序列方向流水搬进来。

        白话""" + ASC_TAG + """：每块查询自己从前往后扫能看见的那一段键值，边搬边把
        这一段的分数与已有结果合起来；搬运只搬有效行，越界的行不会进机器，最后按每行的
        合计份额除一下就是交货内容。
        """,
    ],
    "gdn_impl": [
        """被追踪的那一层：头数、键宽、值宽是常量，时间轴走动态维并全程串行。

        白话""" + CPU_TAG + """：每头守着一块小黑板（状态），每来一条新记录就走五步——
        整片按比例淡掉、照当前键把已有内容读出来当预期、用真值减预期乘上下笔力度补进去、
        最后照查询把黑板读一遍就是这一步的输出。时间轴一步压不得，所以这条链只能串行走。
        """,
        """被追踪的那一层：状态常驻工人手边的小格子，通道轴才交给并行。

        白话""" + ASC_TAG + """：黑板按头各留一份、整个递推期间不落地；每条记录仍是
        "淡掉—召回—补写—读出"五步，只是五步里沿通道的那一维可以同时铺开算。时间轴的先后
        次序不能改，所以外层循环一定是串行。
        """,
    ],
    "gather_impl": [
        """被追踪的那一层：被点名的行数与总行数都是动态维，行宽是常量。

        白话""" + CPU_TAG + """：按点名册从大表里一行行抽出来摆成新表，整行一次搬完；
        册子里重复点同一行就摆两遍，抽出来的数字一个都不改。
        """,
        """被追踪的那一层：抽多少行是活的，搬运按行整段发起。

        白话""" + ASC_TAG + """：册子上的行号直接当搬运的起点用，一次搬一整行到新表对应
        的位置；不够一轮的量就不发起，搬过来的数字原封不动，不做任何算术。
        """,
    ],
    "scatter_impl": [
        """被追踪的那一层：退回的行数是动态维，行宽与目标表行数按调用方给定的形状落定。

        白话""" + CPU_TAG + """：把抽出来的这些行按同一张点名册原路退回大表；同一行退回
        两次就是真的加两次，绝不互相盖掉。目标表在退回前必须已经是清零的。
        """,
        """被追踪的那一层：退回动作按行摊开，累加落在同一张公共表上。

        白话""" + ASC_TAG + """：每个工人手里拿几行往大表对应行上添，同一行被多个人添也
        各自都算数；这一步只做加法，不比较、不覆盖、不做任何换算。
        """,
    ],
    "probe_asc": [
        """昇腾方言的最小自证件：一次搬运、一次逐格乘二、再一次搬回。

        白话：不测任何业务算式，只看这台机器能不能把"搬进来、每人算一格、搬出去"这条路
        编译成产物；能过就说明方言与后端是齐的，后面再谈算子写得对不对。
        """,
    ],
    "probe_cpu": [
        """CPU(c) 后端的最小自证件：与昇腾探针同一套动作，串行逐格乘二。

        白话：先证明"能编译、能真跑、结果逐位对"这三件事在 CPU 上成立，语义对拍才有地基；
        它只当环境体检用，不参与任何业务算式的验收。
        """,
    ],
    "probe": [
        """selfcheck 的方言探针件：一次搬入、一次并行乘二、一次搬出。

        白话：这段只回答"昇腾方言能不能 lowering 成产物"，故意不带任何业务算式，
        这样报出来的失败只可能指向环境，不会被误读成"内核写错了"。
        """,
        """selfcheck 的 CPU 探针件：与昇腾探针同动作，串行逐格乘二。

        白话：用来证明本地这条验收路（CPU 语义对拍）本身是通的——能编译、能发射、
        结果逐位对得上，后面七件的对拍结论才有立足点。
        """,
    ],
}


def _indent_of_first_stmt(node: ast.FunctionDef, lines: list[str]) -> str:
    col = node.body[0].col_offset
    return " " * col


def patch(path: Path) -> int:
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    # 收集"缺 docstring 且不是私有名字"的函数节点，按 (名字, 行号) 排序决定文案取用顺序
    todo: dict[str, list[ast.FunctionDef]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if len([ln for ln in seg.splitlines() if ln.strip()]) <= 4:
            continue
        if (ast.get_docstring(node) or "").strip():
            continue
        todo.setdefault(node.name, []).append(node)
    # 先算好所有插入点，再**自下而上**插：插完一处，后面所有行号都会偏移
    plans = []
    for name, nodes in todo.items():
        nodes.sort(key=lambda x: x.lineno)
        texts = TEXTS[name]
        assert len(texts) >= len(nodes), (path, name, len(texts), len(nodes))
        for node, text in zip(nodes, texts):
            idx = node.body[0].lineno - 1          # 插入点：原第一条语句之前
            pad = _indent_of_first_stmt(node, lines)
            block = "".join(f"{pad}{ln}\n" for ln in text.rstrip("\n").splitlines())
            plans.append((idx, block))
    for idx, block in sorted(plans, reverse=True):
        lines.insert(idx, block)
    n = len(plans)
    path.write_text("".join(lines))
    return n


total = 0
for rel in ["ascend/kernels/gemm_asc.py", "ascend/kernels/gemm_bwd_dw_asc.py",
            "ascend/kernels/rope_asc.py", "ascend/kernels/add_ln_asc.py",
            "ascend/kernels/letter_readout_asc.py", "ascend/kernels/attn_sw_asc.py",
            "ascend/kernels/gdn_asc.py", "ascend/kernels/ascend_env.py",
            "ascend/selfcheck.py"]:
    p = Path(rel)
    before = p.read_text()
    k = patch(p)
    total += k
    import ast as _ast
    _ast.parse(p.read_text())
    print(f"{rel}: +{k} docstring, syntax OK, size {len(before)}->{len(p.read_text())}")
print("TOTAL", total)
