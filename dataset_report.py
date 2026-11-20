"""数据集体检：把题目一的算法跑遍一整个目录，输出 CSV + 汇总。

用法：
    .\\.venv\\Scripts\\python.exe dataset_report.py data\\images
    .\\.venv\\Scripts\\python.exe dataset_report.py data\\images --out out\\report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from task1_color_blocks import IMG_EXTS, find_blocks, order_blocks


def analyze(path: Path, opts: dict | None = None) -> dict | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    t0 = time.perf_counter()
    blocks = order_blocks(find_blocks(bgr, **(opts or {})))
    ms = (time.perf_counter() - t0) * 1000.0
    brightness = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].mean())
    return {
        "file": path.name,
        "width": bgr.shape[1],
        "height": bgr.shape[0],
        "brightness": round(brightness, 1),
        "n_blocks": len(blocks),
        "sequence": "".join(b["name"] for b in blocks),
        "ms": round(ms, 1),
    }


def bucket(brightness: float) -> str:
    if brightness < 70:
        return "暗(V<70)"
    if brightness < 140:
        return "正常(70~140)"
    return "亮(V>=140)"


def main() -> None:
    ap = argparse.ArgumentParser(description="题目一数据集批量体检")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="CSV 输出路径")
    ap.add_argument("--expect", type=int, default=6, help="期望的色块数，默认 6")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总，不打印失败清单")
    ap.add_argument("--fill-min", type=float, default=0.85, help="矩形度阈值")
    args = ap.parse_args()

    opts = dict(fill_min=args.fill_min)

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"{args.folder} 里没找到图片")

    rows: list[dict] = []
    for i, f in enumerate(files, 1):
        r = analyze(f, opts)
        if r is not None:
            rows.append(r)
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(files)}", file=sys.stderr)

    if not rows:
        raise SystemExit("没有可读的图片")

    # ---- 分档统计 ----
    print(f"\n共 {len(rows)} 张，平均耗时 {np.mean([r['ms'] for r in rows]):.1f} ms/张")
    print(f"{'亮度档':<14}{'张数':>5}{'平均块数':>9}{'6块正确率':>11}{'平均耗时':>9}")
    print("-" * 50)
    for name in ("暗(V<70)", "正常(70~140)", "亮(V>=140)"):
        group = [r for r in rows if bucket(r["brightness"]) == name]
        if not group:
            continue
        ok = sum(1 for r in group if r["n_blocks"] == args.expect)
        print(f"{name:<14}{len(group):>5}{np.mean([r['n_blocks'] for r in group]):>9.1f}"
              f"{ok / len(group) * 100:>10.0f}%{np.mean([r['ms'] for r in group]):>8.1f}ms")

    bad = [r for r in rows if r["n_blocks"] != args.expect]
    print(f"\n块数 != {args.expect} 的图片：{len(bad)} 张"
          f"（{len(bad) / len(rows) * 100:.0f}%）")
    if not args.quiet:
        for r in bad[:15]:
            print(f"  {r['file']:<28} {r['n_blocks']} 块  亮度 {r['brightness']:>5}  "
                  f"序列 {r['sequence'] or '（空）'}")

    # ---- CSV ----
    out = args.out or (args.folder / "report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已写入: {out}")

    print("\n下一步建议：")
    print("  1. 把块数 != 6 的图挑出来单独看，按失败原因分类（暗光/倾斜/粘连/反光/白黑块）；")
    print("  2. 每类挑 2~3 张放进报告，配一句原因 —— 这是「失败案例」章节；")
    print("  3. 亮度分档里正确率明显掉的那一档，就是参数该自适应的地方。")


if __name__ == "__main__":
    main()
