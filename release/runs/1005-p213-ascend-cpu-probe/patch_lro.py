from pathlib import Path

p = Path("ascend/kernels/letter_readout_asc.py")
src = p.read_text()

pairs = [
    # 选件表：原来那条三元链在 "ascend + scatter" 组合下会错选 gather 件，改成显式查表
    ('''    impl = (gather_asc_impl if name == ascend_env.TARGET_ASCEND else gather_cpu_impl
            if kind == "gather" else
            scatter_add_asc_impl if name == ascend_env.TARGET_ASCEND else scatter_add_cpu_impl)''',
     '''    # 选件按 (活种, target) 显式查表：三元链在这里极易写反（ascend+scatter 会错选到 gather 件）
    impl = _IMPLS[(kind, name)]'''),
    ('''def _plan(src: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor, dim: int,''',
     '''#: 活种 x target 到方言正文的选件表，四条组合都摆出来，避免表达式里藏分支
_IMPLS = {
    ("gather", ascend_env.TARGET_CPU): gather_cpu_impl,
    ("gather", ascend_env.TARGET_ASCEND): gather_asc_impl,
    ("scatter_add", ascend_env.TARGET_CPU): scatter_add_cpu_impl,
    ("scatter_add", ascend_env.TARGET_ASCEND): scatter_add_asc_impl,
}


def _plan(src: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor, dim: int,'''),
    ("[T.ceildiv(picked, 1)]", "[picked]", 2),
]
for item in pairs:
    old, new, want = item if len(item) == 3 else (item[0], item[1], 1)
    assert src.count(old) == want, (src.count(old), want, old[:50])
    src = src.replace(old, new)
p.write_text(src)
print("patched", len(pairs))
