from pathlib import Path

p = Path("ascend/kernels/letter_readout_asc.py")
src = p.read_text()

pairs = [
    ('''#: 昇腾侧一次发射占用的块数与流水深度
NUM_BLOCKS = 8''',
     '''#: 行号表允许的位宽：内核只收 int32（入口层负责降级），int64 走 torch 回退时也能直接用
ID_DTYPES = (torch.int32, torch.int64)
#: 昇腾侧一次发射占用的块数与流水深度
NUM_BLOCKS = 8'''),
    ('    if ids.dim() != 1 or not ids.dtype.is_integer:',
     '    if ids.dim() != 1 or ids.dtype not in ID_DTYPES:'),
    ('''    if ids.dim() != 1 or not torch.is_tensor(dy) or dy.dim() < 2:
        raise ValueError(f"需要一维行号表与至少二维的 dy，实得 {tuple(ids.shape)} / {tuple(dy.shape)}")''',
     '''    if ids.dim() != 1 or ids.dtype not in ID_DTYPES:
        raise ValueError(f"ids 需是一维整型行号表（int32/int64），实得 {tuple(ids.shape)} / {ids.dtype}")
    if dy.dim() < 2:
        raise ValueError(f"dy 至少二维，实得 {dy.dim()} 维")'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:60])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
