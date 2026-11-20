"""题目一：识别颜色信息板上的色块，按“自上而下”的顺序输出每个色块的颜色。

思路（为什么不是“一次阈值 + findContours”）
----------------------------------------
板上相邻的色块是紧贴着的，如果只做一张高饱和掩膜，红绿蓝会连成一整块，
findContours 只能拿到一个轮廓。所以改成 **按色相分组** 分别做掩膜：
红/橙/黄/绿/青/蓝/紫 各自一张掩膜，不同颜色的块自然分开，
同色的块才会合并（这种情况下再看连通域大小，或改用“板内逐行扫描”方案）。

每个色相里再用三条几何约束过滤掉干扰（地面、花盆、反光）：
    面积占比、长宽比(色块是横条)、矩形度 area/minAreaRect 面积。
最后按 x 聚成列、列内按 y 排序，就是“自上而下”的输出顺序。

用法
----
    python task1_color_blocks.py data/task1/1.jpg
    python task1_color_blocks.py data/task1 --out out/task1 --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from viz import draw_label

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 色相区间（OpenCV 的 H 是 0~179）
HUE_CLASSES: list[tuple[str, list[tuple[int, int]]]] = [
    ("红", [(0, 9), (170, 180)]),
    ("橙", [(9, 22)]),
    ("黄", [(22, 34)]),
    ("绿", [(34, 78)]),
    ("青", [(78, 100)]),
    ("蓝", [(100, 130)]),
    ("紫", [(130, 170)]),
]


# --------------------------------------------------------------------- 颜色判定
def classify_hsv(bgr: np.ndarray) -> str:
    """把一个 BGR 均值颜色映射到颜色名（用于二次确认/输出）。"""
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(h), int(s), int(v)
    if v < 45:
        return "黑"
    if s < 55:
        return "白" if v > 190 else "灰"
    if h < 9 or h >= 170:
        return "红"
    if h < 22:
        return "橙"
    if h < 34:
        return "黄"
    if h < 78:
        return "绿"
    if h < 100:
        return "青"
    if h < 130:
        return "蓝"
    if h < 157:
        return "紫"
    return "红"


# --------------------------------------------------------------------- 色块检测
def find_blocks(bgr: np.ndarray,
                min_area_ratio: float = 2e-3,
                max_area_ratio: float = 0.15,
                aspect_min: float = 1.4,
                fill_min: float = 0.85) -> list[dict]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img_area = float(h * w)

    blocks: list[dict] = []
    for name, ranges in HUE_CLASSES:
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, (lo, 100, 45), (hi, 255, 255))
        if not mask.any():
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]:
            area = cv2.contourArea(cnt)
            if not (min_area_ratio * img_area <= area <= max_area_ratio * img_area):
                continue

            (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
            if rw < rh:                              # 长边始终当作“宽”
                rw, rh, angle = rh, rw, angle + 90.0
            if rh < 1.0 or rw / rh < aspect_min:     # 1.0是一个像素的意思 为了过滤掉很小的噪声
                continue
            if area / (rw * rh) < fill_min:          # 矩形度不够 = 不是规整色块
                continue

            box = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.int32)

            # 只统计中心 40% 的像素，避开边缘混色/抗锯齿
            patch = np.zeros((h, w), np.uint8)
            inner = cv2.boxPoints(((cx, cy), (rw * 0.4, rh * 0.4), angle))
            cv2.fillPoly(patch, [inner.astype(np.int32)], 255)
            mean_bgr = np.array(cv2.mean(bgr, mask=patch)[:3])

            blocks.append({
                "center": (float(cx), float(cy)),
                "box": box,
                "size": (float(rw), float(rh)),
                "bgr": mean_bgr,
                "name": classify_hsv(mean_bgr),
                "hue_class": name,
            })
    return blocks


def order_blocks(blocks: list[dict]) -> list[dict]:
    """先按 x 聚成列（相邻块中心间距 > 1.5 倍块宽就断列），列内按 y 从上到下。"""
    if not blocks:
        return []
    by_x = sorted(blocks, key=lambda b: b["center"][0])
    med_w = float(np.median([b["size"][0] for b in by_x]))
    gap = max(med_w * 1.5, 25.0)

    columns: list[list[dict]] = [[by_x[0]]]
    for b in by_x[1:]:
        if b["center"][0] - columns[-1][-1]["center"][0] > gap:
            columns.append([b])
        else:
            columns[-1].append(b)

    ordered: list[dict] = []
    for col in columns:
        col.sort(key=lambda b: b["center"][1])
        ordered.extend(col)
    return ordered


# --------------------------------------------------------------------- 可视化
def annotate(bgr: np.ndarray, ordered: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for i, b in enumerate(ordered, 1):
        cv2.polylines(out, [b["box"]], True, (255, 255, 255), 2, cv2.LINE_AA)
        cx, cy = int(b["center"][0]), int(b["center"][1])
        label = f"{i}.{b['name']}"
        out = draw_label(out, label, cx - 34, cy - 14,
                         fill=(255, 255, 255), text_color=(0, 0, 0), font_size=20)
    return out


# --------------------------------------------------------------------- 主流程
def process_one(path: Path, out_dir: Path | None, debug: bool,
                opts: dict) -> list[str]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"[跳过] 读不到图片: {path}")
        return []

    ordered = order_blocks(find_blocks(bgr, **opts))

    print(f"\n图片: {path.name}   找到 {len(ordered)} 个色块")
    for i, b in enumerate(ordered, 1):
        bgr_str = ", ".join(f"{v:5.1f}" for v in b["bgr"])
        print(f"  {i:>2}. {b['name']}  BGR=({bgr_str})  "
              f"中心=({b['center'][0]:.0f},{b['center'][1]:.0f})  "
              f"尺寸={b['size'][0]:.0f}x{b['size'][1]:.0f}")
    names = [b["name"] for b in ordered]
    print("自上而下顺序输出: " + " ".join(names))

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{path.stem}_result.jpg"), annotate(bgr, ordered))
        if debug:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            union = np.zeros(bgr.shape[:2], np.uint8)
            for _, ranges in HUE_CLASSES:
                for lo, hi in ranges:
                    union |= cv2.inRange(hsv, (lo, 90, 45), (hi, 255, 255))
            cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), union)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="颜色信息板色块识别")
    ap.add_argument("inputs", nargs="+", type=Path, help="图片文件或目录")
    ap.add_argument("--out", type=Path, default=None, help="结果图保存目录")
    ap.add_argument("--debug", action="store_true", help="额外保存 HSV 掩膜")
    ap.add_argument("--min-area-ratio", type=float, default=2e-3)
    ap.add_argument("--max-area-ratio", type=float, default=0.15)
    ap.add_argument("--aspect-min", type=float, default=1.4,
                    help="色块宽/高下限，块越“方”就调小")
    ap.add_argument("--fill-min", type=float, default=0.85,
                    help="轮廓面积/外接矩形面积 下限")
    args = ap.parse_args()

    opts = dict(min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                aspect_min=args.aspect_min,
                fill_min=args.fill_min)

    files: list[Path] = []
    for p in args.inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        elif p.is_file():
            files.append(p)

    for f in files:
        process_one(f, args.out, args.debug, opts)


if __name__ == "__main__":
    main()
