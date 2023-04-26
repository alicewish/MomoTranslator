import re

import cv2
from loguru import logger
from PIL import Image, ImageDraw
import numpy as np

from core.shape import Rect
from storage import GlobalStorage


def get_frame_grid_recursive(rect, grids, frame_mask, media_type):
    rec = Rect(rect, media_type)
    rec.get_devide(frame_mask)

    if rec.pg.basic:
        grids.append(rec)
    else:
        rects = rec.pg.rects
        for r in rects:
            get_frame_grid_recursive(r, grids, frame_mask, media_type)

@logger.catch
def get_marked_frames(frame_grid_strs, image_raw):
    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    for f in range(len(frame_grid_strs)):
        frame_grid_str = frame_grid_strs[f]
        # 使用正则表达式提取字符串中的所有整数
        int_values = list(map(int, re.findall(r'\d+', frame_grid_str)))
        # 按顺序分配值
        x, y, w, h, xx, yy, ww, hh = int_values

        pt1 = (x, y)
        pt2 = (x + w, y + h)
        draw.rectangle([pt1, pt2], outline=(255, 255, 0, 128), width=3)
        pt1 = (xx, yy)
        pt2 = (xx + ww, yy + hh)
        draw.rectangle([pt1, pt2], outline=(0, 255, 255, 128), width=3)

        # Draw the index of the rectangle
        index_position = (x + 5, y - 5)
        draw.text(index_position, str(f + 1), fill=(0, 0, 255, 255), font=GlobalStorage.font60)

    marked_frames = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return marked_frames

def compute_frame_mask(image_raw, dominant_color, tolerance=5):
    dominant_color_int32 = dominant_color[:3].astype(np.int32)
    lower_bound = np.maximum(0, dominant_color_int32 - tolerance)
    upper_bound = np.minimum(255, dominant_color_int32 + tolerance)

    # Bugfix: unlike jpeg format images, imdecode obtains rgba quads for png format
    image = image_raw
    if image_raw.shape[2] == 4:
        image = np.delete(image_raw, 3, axis=2)

    mask = np.inRange(image, lower_bound, upper_bound)
    return mask
    