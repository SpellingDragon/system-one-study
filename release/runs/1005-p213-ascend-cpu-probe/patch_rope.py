from pathlib import Path

p = Path("ascend/kernels/rope_asc.py")
src = p.read_text()
old = """    for slot in rotate_slots:
        x1 = qkv[:, slot, :, :half].float()
        x2 = qkv[:, slot, :, half:].float()
        qkv[:, slot, :, :half] = (x1 * c - x2 * s).to(qkv.dtype)
        qkv[:, slot, :, half:] = (x2 * c + x1 * s).to(qkv.dtype)
"""
new = """    for slot in rotate_slots:
        # x1/x2 必须先各留一份原件：qkv 已是 fp32 时 .float() 返回的是视图，
        # 直接写前半会把 x1 改掉，后半那一行就会读到被污染的值（就地语义下的经典自噬）。
        x1 = qkv[:, slot, :, :half].float().clone()
        x2 = qkv[:, slot, :, half:].float().clone()
        qkv[:, slot, :, :half] = (x1 * c - x2 * s).to(qkv.dtype)
        qkv[:, slot, :, half:] = (x2 * c + x1 * s).to(qkv.dtype)
"""
assert old in src, "pattern missing"
p.write_text(src.replace(old, new, 1))
print("patched", new.count("\n"))
