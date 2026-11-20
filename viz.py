"""画框辅助：给视频/图片加“带底色的文字标签”。

cv2.putText 只支持 ASCII，中文字符会变成 `?`，所以这里用 Pillow 兜底。
英文/数字走 cv2 快速路径，中文走 Pillow，两边都返回 BGR ndarray。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_font_cache: dict[int, object] = {}


def _font(size: int):
    if size not in _font_cache:
        from PIL import ImageFont
        for p in FONT_CANDIDATES:
            if Path(p).exists():
                try:
                    _font_cache[size] = ImageFont.truetype(p, size)
                    break
                except OSError:
                    continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_label(img: np.ndarray, text: str, x: int, y: int,
               fill=(0, 255, 0), text_color=(0, 0, 0),
               font_size: int = 20) -> np.ndarray:
    """在 (x, y) 左上角画一个实心标签。y 通常取框的上沿再往上挪一点。

    fill / text_color 都是 BGR。
    """
    x, y = int(x), int(y)
    if text.isascii():
        scale = font_size / 30.0
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         scale, 2)
        y = max(0, y)
        cv2.rectangle(img, (x, y), (x + tw + 6, y + th + base + 4), fill, -1)
        cv2.putText(img, text, (x + 3, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, text_color, 2, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    y = max(0, y)
    draw.rectangle([x, y, x + tw + 8, y + th + 10],
                   fill=tuple(int(c) for c in fill[::-1]))
    draw.text((x + 4, y + 2 - bbox[1]), text, font=font,
              fill=tuple(int(c) for c in text_color[::-1]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
