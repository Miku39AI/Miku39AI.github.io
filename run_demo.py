# -*- coding: utf-8 -*-
"""演示：规整抽样重组 + 用「每行块数」折行出 2/4 个人"""
from PIL import Image
from tile_reassemble import cut_and_reassemble

# 从参考截图右侧取一块干净画面，放大 2 倍当演示素材（460x624）
src = Image.open("input.jpg").crop((1119, 392, 1349, 704)).resize((460, 624), Image.LANCZOS)
src.save("demo_src.png")

BASE = dict(tile_w=46, tile_h=39, stride=(2, 2), hole_mode="marker",
            marker_ratio=0.4, panel_gap=4)

# 网格：10x16 块，抽样后 5x8 = 40 块
out1, panel1, _ = cut_and_reassemble(src, **BASE, panel_cols=5)
out1.save("demo_auto.png")

# 每行 10 块 = 2 × 抽取列数(5) → 折出 2 列，一个人变成两个人
out2, panel2, _ = cut_and_reassemble(src, **BASE, panel_cols=10)
out2.save("demo_2people.png")

# 每行 20 块 = 4 × 抽取列数 → 4 个人
out4, panel4, _ = cut_and_reassemble(src, **BASE, panel_cols=20)
out4.save("demo_4people.png")

print("demo files written")
