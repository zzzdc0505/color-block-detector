"""单张诊断：打印每个检出块的细节，并把编号图画出来。

    .\\.venv\\Scripts\\python.exe debug_one.py "images\\xxx.jpg" --out out\\dbg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from task1_color_blocks import annotate, find_blocks, order_blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/dbg"))
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"读不到 {args.image}")

    blocks = find_blocks(img)
    print(f"图片 {args.image.name}  {img.shape[1]}x{img.shape[0]}")
    print(f"检出 {len(blocks)} 块（按 y 排序展示）：")
    print(f"{'y':>6} {'x':>6} {'尺寸':>11} {'BGR':>22}  判定     色相类")
    for b in sorted(blocks, key=lambda b: b["center"][1]):
        print(f"{b['center'][1]:6.0f} {b['center'][0]:6.0f} "
              f"{b['size'][0]:5.0f}x{b['size'][1]:<5.0f} "
              f"({b['bgr'][0]:5.1f},{b['bgr'][1]:5.1f},{b['bgr'][2]:5.1f})  "
              f"{b['name']:<4}({b['hue_class']})")

    ordered = order_blocks(blocks)
    print("\norder_blocks 输出:", " ".join(b["name"] for b in ordered))

    args.out.mkdir(parents=True, exist_ok=True)
    vis = annotate(img, ordered)
    vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                     interpolation=cv2.INTER_AREA)
    p = args.out / f"{args.image.stem}_dbg.jpg"
    cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"标注图: {p}")


if __name__ == "__main__":
    main()
