#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切图 + 规整抽样重组（用较少的块还原原图）
==========================================
把图片切成规则网格，然后“规整抽取”（像降采样一样，横向、纵向每隔固定步长取一块），
把抽出的块**按原位置顺序**重新拼成一张小图——这张小图就是原图的缩略版。
例如步长 stride=2 时只用 1/4 的块，拼出来的图刚好是原图的 1/2 大小，
却能完整还原原图的轮廓；stride=3 时用 1/9 的块拼出 1/3 大小。

做法分四步：
  1. 网格切割：cols × rows 个 tile_w × tile_h 的块；
  2. 规整抽取：行号 % stride_y == 0 且 列号 % stride_x == 0 的块被选中；
  3. 原图留白：被抽走的位置画白色方块（整格留白，或参考图那种左上角小白块标记）；
  4. 有序重组：抽出的块按“行优先”原顺序拼到右侧小图（可选块间白缝、整体缩放）。

用法
----
    python tile_reassemble.py 输入图.jpg
    python tile_reassemble.py photo.png --stride 3 3        # 用 1/9 的块
    python tile_reassemble.py photo.png --stride 2 1        # 只横向抽稀
    python tile_reassemble.py photo.png --hole-mode full    # 整格留白
    python tile_reassemble.py photo.png --scramble          # 乱序（对照用）
    python tile_reassemble.py photo.png --panel-cols 20     # 每行 20 块（折行重排）
    python tile_reassemble.py photo.png --people 2          # 一个人 → 两个人

参数
----
--tile W H        单个块的大小（像素），默认 48 60
--stride SX SY    抽取步长：横向/纵向每隔几块取一块，默认 2 2（取 1/4 的块）
--hole-mode MODE  full   = 被抽走的位置整格填白
                  marker = 参考图样式：该格左上角画一个白色小块
--marker-ratio R  marker 模式下小白块占格子宽高的比例（默认 0.4）
--panel-gap G     重组区相邻块之间的白色缝隙（默认 4 像素）
--panel-cols N    重组区每行块数（默认 = 抽取列数，正好拼成缩略图）；
                  设为 K×抽取列数 时，整条块流按 K 倍宽度折行，画面被折出 K 列
                  —— 这就是“一个人变成 K 个人”的效果来源
--people K        便捷写法：等价于 --panel-cols K×抽取列数（K=2/4/8 效果最好）
--shrink F        重组时把每块再缩放 F 倍（默认 1 不缩放）
--scramble        乱序重组（默认按原顺序，才能还原原图样子）
--bg R G B        背景/留白颜色（默认 255 255 255）
--out PATH        输出文件路径
--seed N          随机种子（仅 --scramble 时有效）
"""
import argparse
import math
import random

import numpy as np
from PIL import Image, ImageDraw


def _seamless_panel(src, order, cols, pc, pr, tw, th, ov, tone, shrink, bg_color):
    """羽化融合拼接：每块向右/下多带 ov 像素内容，用 smoothstep 权重过渡，
    相邻块的边缘在重叠区内相互融合，接缝就不再是硬边。"""
    arr = np.asarray(src.convert("RGB")).astype(np.float32)
    H, W = pr * th, pc * tw
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)

    # 全局均值（用于色调统一，降低块与块之间的色差跳变）
    means = []
    for idx in order:
        sx, sy = (idx % cols) * tw, (idx // cols) * th
        means.append(arr[sy:sy + th, sx:sx + tw].reshape(-1, 3).mean(0))
    gmean = np.mean(means, axis=0)

    for j, idx in enumerate(order):
        c, r = j % pc, j // pc
        if r >= pr:
            break
        sx, sy = (idx % cols) * tw, (idx // cols) * th
        ext_x = ov if c < pc - 1 else 0     # 最后一列不再向外扩，避免白边
        ext_y = ov if r < pr - 1 else 0
        y1 = min(sy + th + ext_y, arr.shape[0])
        x1 = min(sx + tw + ext_x, arr.shape[1])
        block = arr[sy:y1, sx:x1]
        if tone > 0:
            block = block + tone * (gmean - block.reshape(-1, 3).mean(0))

        h_, w_ = block.shape[:2]
        wx = np.ones(w_, np.float32)
        if w_ > tw:                          # 右边缘 ov 像素平滑淡出
            t = np.linspace(0.0, 1.0, w_ - tw, dtype=np.float32)
            wx[tw:] = 1.0 - (t * t * (3 - 2 * t))
        wy = np.ones(h_, np.float32)
        if h_ > th:                          # 下边缘同理
            t = np.linspace(0.0, 1.0, h_ - th, dtype=np.float32)
            wy[th:] = 1.0 - (t * t * (3 - 2 * t))
        mask = (wy[:, None] * wx[None, :])[..., None]

        px, py = c * tw, r * th
        pw, ph = min(w_, W - px), min(h_, H - py)
        acc[py:py + ph, px:px + pw] += block[:ph, :pw] * mask[:ph, :pw]
        wsum[py:py + ph, px:px + pw] += mask[:ph, :pw]

    out = Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8))
    if shrink != 1.0:
        out = out.resize((max(1, int(out.width * shrink)), max(1, int(out.height * shrink))),
                         Image.LANCZOS)
    return out


def _dp_cut(cost, lam=2.0, max_step=3):
    """动态规划找最小代价切割路径。
    cost: (n, m) 每行(列)选 m 中某个位置的代价；返回长度 n 的索引数组。"""
    n, m = cost.shape
    acc = cost[0].astype(np.float64).copy()
    back = np.zeros((n, m), np.int32)
    for y in range(1, n):
        prev = acc
        cur = np.empty(m)
        for x in range(m):
            lo, hi = max(0, x - max_step), min(m, x + max_step + 1)
            vals = prev[lo:hi] + lam * np.abs(np.arange(lo, hi) - x)
            k = int(np.argmin(vals))
            cur[x] = vals[k]
            back[y, x] = lo + k
        acc = cur + cost[y]
    path = np.zeros(n, np.int32)
    path[-1] = int(np.argmin(acc))
    for y in range(n - 1, 0, -1):
        path[y - 1] = back[y, path[y]]
    return path


def _quilt_panel(src, order, cols, pc, pr, tw, th, ov, shrink, bg_color):
    """最小误差缝拼接（quilt / seam cut）：在重叠带里用最小代价路径选择
    “从哪一列开始显示下一块”，让接缝沿着内容最接近的位置走，接缝几乎看不见。"""
    arr = np.asarray(src.convert("RGB")).astype(np.float32)
    h2, w2 = arr.shape[:2]

    def get_tile(idx, w_need, h_need):
        sx, sy = (idx % cols) * tw, (idx // cols) * th
        blk = arr[sy:min(sy + h_need, h2), sx:min(sx + w_need, w2)]
        py, px = h_need - blk.shape[0], w_need - blk.shape[1]
        if py > 0 or px > 0:
            blk = np.pad(blk, ((0, max(0, py)), (0, max(0, px)), (0, 0)), mode="edge")
        return blk

    # ---- 逐行拼条带 ----
    strips = []
    for r in range(pr):
        row = order[r * pc:(r + 1) * pc]
        if not row:
            break
        n = len(row)
        strip = np.zeros((th + ov, n * tw + ov, 3), np.float32)
        strip[:, 0:tw + ov] = get_tile(row[0], tw + ov, th + ov)[:, :tw + ov]
        for c in range(1, n):
            t = get_tile(row[c], tw + ov, th + ov)
            x0 = c * tw
            bandL = strip[:, x0:x0 + ov]          # 上一块向右的延伸
            bandR = t[:, 0:ov]                    # 本块左边
            path = _dp_cut(np.abs(bandL - bandR).sum(axis=2), lam=2.0, max_step=ov - 1)
            fs = 3.0   # 切缝两侧再羽化几个像素，进一步藏住硬线
            ramp = np.linspace(0.0, 1.0, max(2, int(fs * 2)))
            for y in range(th + ov):
                b = int(path[y])
                lo, hi = max(0, b - int(fs)), min(ov, b + int(fs))
                strip[y, x0:x0 + lo] = bandL[y, :lo]
                if hi > lo:
                    wgt = np.linspace(0.0, 1.0, hi - lo)[:, None]
                    strip[y, x0 + lo:x0 + hi] = bandL[y, lo:hi] * (1 - wgt) + bandR[y, lo:hi] * wgt
                strip[y, x0 + hi:x0 + ov] = bandR[y, hi:]
            strip[:, x0 + ov:x0 + tw + ov] = t[:, ov:tw + ov]
        strips.append(strip)

    # ---- 逐条带纵向拼接 ----
    H = pr * th + ov
    W = pc * tw + ov
    canvas = np.full((H, W, 3), 255.0, np.float32)
    canvas[0:th + ov, 0:strips[0].shape[1]] = strips[0]
    for r in range(1, len(strips)):
        s = strips[r]
        y0 = r * th
        w = min(W, s.shape[1])
        bandL = canvas[y0:y0 + ov, :w]
        bandR = s[0:ov, :w]
        path = _dp_cut(np.abs(bandL - bandR).sum(axis=2).T, lam=1.5, max_step=ov - 1)
        fs = 3.0
        for x in range(w):
            b = int(path[x])
            lo, hi = max(0, b - int(fs)), min(ov, b + int(fs))
            canvas[y0:y0 + lo, x] = bandL[:lo, x]
            if hi > lo:
                wgt = np.linspace(0.0, 1.0, hi - lo)[:, None]
                canvas[y0 + lo:y0 + hi, x] = bandL[lo:hi, x] * (1 - wgt) + bandR[lo:hi, x] * wgt
            canvas[y0 + hi:y0 + ov, x] = bandR[hi:, x]
        canvas[y0 + ov:y0 + th + ov, :w] = s[ov:th + ov, :w]

    img = Image.fromarray(np.clip(canvas[:pr * th, :pc * tw], 0, 255).astype(np.uint8))
    if shrink != 1.0:
        img = img.resize((max(1, int(img.width * shrink)), max(1, int(img.height * shrink))),
                         Image.LANCZOS)
    return img


def regular_pick(cols, rows, stride_x, stride_y):
    """规整抽样：返回被选中格子的行优先索引列表（保持原顺序）"""
    picked = []
    for r in range(0, rows, stride_y):
        for c in range(0, cols, stride_x):
            picked.append(r * cols + c)
    return picked


# 正则重排“换行”参数：pc = people × 抽取列数，会把整条块流按 width 折行，
# 于是画面在横向上被“折”出 people 列 —— 一个人变成 N 个人就是这么来的。
def cut_and_reassemble(
    src: Image.Image,
    tile_w: int = 48,
    tile_h: int = 60,
    stride: tuple = (2, 2),
    scramble: bool = False,
    hole_mode: str = "marker",
    marker_ratio: float = 0.4,
    panel_gap: int = 4,
    panel_cols: int | None = None,
    shrink: float = 1.0,
    seamless: int = 0,
    seam_cut: bool = False,
    tone_match: float = 0.0,
    bg_color=(255, 255, 255),
    margin: int = 20,
    seed: int | None = None,
):
    """返回 (合成图, 重组小图, 带洞原图)"""
    if seed is not None:
        random.seed(seed)

    src = src.convert("RGB")
    w, h = src.size
    stride_x, stride_y = stride

    # ---- 1. 网格切割 ----
    cols = max(1, w // tile_w)
    rows = max(1, h // tile_h)
    src = src.crop((0, 0, cols * tile_w, rows * tile_h))
    picked = regular_pick(cols, rows, stride_x, stride_y)

    # ---- 2. 原图留白（被抽走的格子画白色方块） ----
    orig = src.copy()
    draw = ImageDraw.Draw(orig)
    for i in picked:
        r, c = divmod(i, cols)
        x0, y0 = c * tile_w, r * tile_h
        if hole_mode == "full":
            box = (x0, y0, x0 + tile_w, y0 + tile_h)          # 整格填白
        else:
            mw = max(2, int(tile_w * marker_ratio))
            mh = max(2, int(tile_h * marker_ratio))
            box = (x0, y0, x0 + mw, y0 + mh)                  # 左上角小白块
        draw.rectangle(box, fill=bg_color)

    # ---- 3. 抽出的块（按原顺序）重组 ----
    order = picked[:]
    if scramble:
        random.shuffle(order)

    pick_cols = len(range(0, cols, stride_x))  # 抽取的列数
    pc = panel_cols if panel_cols else pick_cols
    pr = max(1, math.ceil(len(order) / pc))

    if seamless and seamless > 0:
        if seam_cut:
            # 无缝模式（拼接法）：沿最小误差路径切割，接缝藏在内容最接近的地方
            panel = _quilt_panel(src, order, cols, pc, pr, tile_w, tile_h,
                                 int(seamless), shrink, bg_color)
        else:
            # 无缝模式（羽化法）：块间平滑过渡
            panel = _seamless_panel(src, order, cols, pc, pr, tile_w, tile_h,
                                    int(seamless), tone_match, shrink, bg_color)
    else:
        tiles = [
            src.crop(((i % cols) * tile_w, (i // cols) * tile_h,
                      (i % cols + 1) * tile_w, (i // cols + 1) * tile_h))
            for i in order
        ]
        if shrink != 1.0:  # 每块再整体缩放
            sw, sh = max(1, int(tile_w * shrink)), max(1, int(tile_h * shrink))
            tiles = [t.resize((sw, sh), Image.BILINEAR) for t in tiles]
        else:
            sw, sh = tile_w, tile_h

        # 拼出“重组图”：整条块流按 pc 块一行折行，pc 不同 → 折出来的画面结构不同
        panel = Image.new(
            "RGB",
            (pc * sw + (pc + 1) * panel_gap, pr * sh + (pr + 1) * panel_gap),
            bg_color,
        )
        for j, t in enumerate(tiles):
            px = panel_gap + (j % pc) * (sw + panel_gap)
            py = panel_gap + (j // pc) * (sh + panel_gap)
            panel.paste(t, (px, py))

    # ---- 4. 合成：左原图 + 右重组图 ----
    out_w = orig.width + margin + panel.width + margin
    out_h = max(orig.height, panel.height) + margin * 2
    out = Image.new("RGB", (out_w, out_h), bg_color)
    out.paste(orig, (margin, margin))
    out.paste(panel, (margin + orig.width + margin, (out_h - panel.height) // 2))

    return out, panel, orig


def main():
    ap = argparse.ArgumentParser(description="切图 + 规整抽样重组（用较少的块还原原图）")
    ap.add_argument("input")
    ap.add_argument("--tile", nargs=2, type=int, default=[48, 60], metavar=("W", "H"))
    ap.add_argument("--stride", nargs=2, type=int, default=[2, 2], metavar=("SX", "SY"))
    ap.add_argument("--hole-mode", choices=["full", "marker"], default="marker")
    ap.add_argument("--marker-ratio", type=float, default=0.4)
    ap.add_argument("--panel-gap", type=int, default=4)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--panel-cols", type=int, default=None,
                    help="重组区每行块数；设为 K×抽取列数 就会折出 K 个人")
    ap.add_argument("--people", type=int, default=None,
                    help="便捷写法：折出 K 个人（=把 panel-cols 设为 K×抽取列数）")
    ap.add_argument("--seamless", type=int, default=0, metavar="PX",
                    help="无缝融合：羽化重叠宽度（像素），0=关闭")
    ap.add_argument("--seam-cut", action="store_true",
                    help="无缝模式改用「最小误差缝」拼接（接缝几乎不可见）")
    ap.add_argument("--tone", type=float, default=0.0,
                    help="色调统一强度 0~1（无缝模式下调低块间色差）")
    ap.add_argument("--scramble", action="store_true")
    ap.add_argument("--bg", nargs=3, type=int, default=[255, 255, 255])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    panel_cols = args.panel_cols
    if panel_cols is None and args.people:
        img = Image.open(args.input)
        cols = max(1, img.size[0] // args.tile[0])
        pick_cols = len(range(0, cols, args.stride[0]))
        panel_cols = args.people * pick_cols

    out, panel, orig = cut_and_reassemble(
        Image.open(args.input),
        tile_w=args.tile[0],
        tile_h=args.tile[1],
        stride=tuple(args.stride),
        scramble=args.scramble,
        hole_mode=args.hole_mode,
        marker_ratio=args.marker_ratio,
        panel_gap=args.panel_gap,
        panel_cols=panel_cols,
        shrink=args.shrink,
        seamless=args.seamless,
        tone_match=args.tone,
        bg_color=tuple(args.bg),
        seed=args.seed,
    )
    dst = args.out or "result.png"
    out.save(dst)
    print(f"saved -> {dst} ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
