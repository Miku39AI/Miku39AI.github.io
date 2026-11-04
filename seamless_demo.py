# -*- coding: utf-8 -*-
"""无缝融合 Demo：硬接缝 / 羽化融合 / 羽化+色调统一 / 最小误差缝"""
import os
from PIL import Image, ImageDraw, ImageFont
from tile_reassemble import cut_and_reassemble

SRC = "demo_src.png"          # 460x624
PC = 10                       # 每行 10 块 = 2 × 抽取列数 → 折出 2 列
BASE = dict(tile_w=46, tile_h=39, stride=(2, 2), panel_cols=PC,
            hole_mode="marker", marker_ratio=0.4, panel_gap=0)

variants = [
    ("A 硬接缝", dict(seamless=0), "seam_demo_A_hard.png"),
    ("B 羽化 8px", dict(seamless=8), "seam_demo_B_feather8.png"),
    ("C 羽化14+色调", dict(seamless=14, tone_match=0.35), "seam_demo_C_feather14_tone.png"),
    ("D 最小误差缝", dict(seamless=10, seam_cut=True), "seam_demo_D_quilt.png"),
]

panels = []
for label, kw, fn in variants:
    _, panel, _ = cut_and_reassemble(Image.open(SRC), **BASE, **kw)
    panel.save(fn)
    panels.append((label, panel))
    print("saved", fn, panel.size)

FONT = None
for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(f):
        FONT = ImageFont.truetype(f, 16)
        break


def board(items, zoom=None, out="board.png"):
    imgs = []
    for label, im in items:
        z = im.crop((0, 0, 210, 130)) if zoom else im
        if zoom:
            z = z.resize((z.width * 2, z.height * 2), Image.LANCZOS)
        imgs.append((label, z))
    w = sum(im.width for _, im in imgs) + 14 * (len(imgs) + 1)
    h = max(im.height for _, im in imgs) + 34 + 14
    b = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(b)
    x = 14
    for label, im in imgs:
        b.paste(im, (x, 34))
        d.text((x + 2, 9), label, fill=(40, 40, 40), font=FONT)
        d.rectangle([x - 1, 33, x + im.width, 34 + im.height], outline=(205, 205, 205))
        x += im.width + 14
    b.save(out)
    print("saved", out, b.size)


board(panels, zoom=True, out="seam_compare_zoom.png")
board(panels, zoom=None, out="seam_compare_full.png")
