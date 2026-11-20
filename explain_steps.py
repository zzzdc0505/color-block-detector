"""生成四格流程图：原图 -> 高饱和掩膜 -> 分色相候选 -> 最终结果。

    python explain_steps.py demo/board.png --out out/demo/pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from task1_color_blocks import HUE_CLASSES, annotate, find_blocks, order_blocks
from viz import draw_label

REP_BGR = {
    "红": (40, 40, 210), "橙": (0, 120, 230), "黄": (0, 210, 225),
    "绿": (40, 180, 60), "青": (200, 200, 0), "蓝": (200, 80, 0),
    "紫": (200, 40, 160),
}


def caption(img: np.ndarray, text: str) -> np.ndarray:
    return draw_label(img, text, 6, 6, fill=(255, 255, 255),
                      text_color=(0, 0, 0), font_size=22)


def build(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # ① 原图
    p1 = caption(bgr.copy(), "1. 输入")

    # ② 一张“高饱和”掩膜：注意 6 个色块粘成了一大片
    single = cv2.inRange(hsv, (0, 90, 45), (180, 255, 255))
    single = cv2.morphologyEx(single, cv2.MORPH_CLOSE, kernel)
    p2 = cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)
    num, _, stats, _ = cv2.connectedComponentsWithStats(single)
    big = int((stats[1:, cv2.CC_STAT_AREA] >= 0.002 * h * w).sum())
    p2 = caption(p2, f"2. 一张掩膜：6 块粘成 {big} 个连通域")

    # ③ 按色相分开
    p3 = np.zeros_like(bgr)
    for name, ranges in HUE_CLASSES:
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, (lo, 90, 45), (hi, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        p3[mask > 0] = REP_BGR[name]
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]:
            if cv2.contourArea(cnt) > 0.002 * h * w:
                cv2.drawContours(p3, [cnt], -1, (255, 255, 255), 2)
    p3 = caption(p3, "3. 按色相分组：每块独立（黄=地面，橙红=花盆）")

    # ④ 几何过滤 + 排序后的结果
    p4 = caption(annotate(bgr, order_blocks(find_blocks(bgr))),
                 "4. 过滤干扰 + 自上而下排序")

    gap = 8
    top = np.hstack([p1, np.full((h, gap, 3), 255, np.uint8), p2])
    bottom = np.hstack([p3, np.full((h, gap, 3), 255, np.uint8), p4])
    canvas = np.vstack([top, np.full((gap, top.shape[1], 3), 255, np.uint8), bottom])
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, nargs="?", default=Path("demo/board.png"))
    ap.add_argument("--out", type=Path, default=Path("out/demo/pipeline.png"))
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"读不到图片: {args.image}")
    canvas = build(bgr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"已保存: {args.out}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
