import os
from pathlib import Path
from shutil import copy2

import cv2
from natsort import natsorted
import numpy as np
from PIL import Image

from helper import timer_decorator
from utils import md5

tags = ('加框','分框','框','涂白','填字','修图','-','copy','副本','拷贝','顺序','打码','测试','标注','边缘','标志','伪造')

# ================Rewrite================

def collect_raw_images_from(rootdir, recursive=True):

    import imghdr
    images = {}

    if rootdir:
        if not recursive:
            # To avoid duplicate sorting
            sorted_file_list = natsorted(os.listdir(rootdir)) 
            for file_name in sorted_file_list:
                file_path = Path(os.path.join(rootdir, file_name))
                if os.path.isfile(file_path):
                    file_type = imghdr.what(file_path)
                    if file_type == 'jpeg':
                        if 'jpg' not in images:
                            images['jpg'] = []
                        images['jpg'].append(file_path)
                    elif file_type == 'png':
                        if 'png' not in images:
                            images['png'] = []
                        images['png'].append(file_path)
                    # TODO: support for other image formats
                    else:
                        pass
                # do nothing
                else:
                    pass
        else:
            for rootdir, dir_names, file_names in os.walk(rootdir):
                for file_name in file_names:
                    file_path = Path(os.path.join(rootdir, file_name))
                    if os.path.isfile(file_path):
                        file_type = imghdr.what(file_path)
                        if file_type == 'jpeg':
                            if 'jpg' not in images:
                                images['jpg'] = []
                            images['jpg'].append(file_path)
                        elif file_type == 'png':
                            if 'png' not in images:
                                images['png'] = []
                            images['png'].append(file_path)
                        # TODO: support for other image formats
                        else:
                            pass
                    # do nothing
                    else:
                        pass

    # do nothing or raise exception
    else:
        pass

    return images

def extract_mask_from(file_paths):
    all_masks = []
    no_masks = []

    for file_path in file_paths:
        if '-Mask-' in file_path.stem:
            all_masks.append(file_path)
        else:
            no_masks.append(file_path)

    return all_masks, no_masks

def block_tagged_by(file_paths, tag, mode="stem"):
    if mode == 'name':
        pure_paths = [x for x in file_paths if not x.name.endswith(tag)]
    else:
        pure_paths = [x for x in file_paths if not x.stem.endswith(tag)]

    return pure_paths


def collect_valid_images_from(rootdir, mode='raw'):
    images = collect_raw_images_from(rootdir)

    if 'jpg' in images and mode == 'raw':
        return block_tagged_by(images['jpg'], tags)
    elif 'png' in images:
        all_masks, no_masks = extract_mask_from(images['png'])
        if mode == 'raw':
            return block_tagged_by(no_masks, tags)
        else:
            return all_masks
    else:
        return []

def write_image(pic_path, picimg):
    pic_path = Path(pic_path)
    ext = pic_path.suffix
    temp_pic = pic_path.parent / f'temp{ext}'

    # 检查输入图像的类型，如果是PIL图像，则将其转换为NumPy数组
    if isinstance(picimg, Image.Image):
        picimg = np.array(picimg)

    # 保存临时图像
    cv2.imencode(ext, picimg)[1].tofile(temp_pic)

    # 检查临时图像和目标图像的md5哈希和大小是否相同
    if not pic_path.exists() or md5(temp_pic) != md5(pic_path):
        copy2(temp_pic, pic_path)

    # 删除临时图像
    if temp_pic.exists():
        os.remove(temp_pic)

    return pic_path

@timer_decorator
def display_projection_pil(mask_image):
    # 将 NumPy 数组转换为 PIL 图像
    mask_image_pil = Image.fromarray(mask_image)

    # 计算水平投影
    horizontal_projection = [row.count(0) for row in mask_image.tolist()]

    # 创建水平投影图像
    horizontal_projection_image = Image.new('L', mask_image_pil.size, 255)
    for y, value in enumerate(horizontal_projection):
        for x in range(value):
            horizontal_projection_image.putpixel((x, y), 0)

    # 计算垂直投影
    vertical_projection = [col.count(0) for col in zip(*mask_image.tolist())]

    # 创建垂直投影图像
    vertical_projection_image = Image.new('L', mask_image_pil.size, 255)
    for x, value in enumerate(vertical_projection):
        for y in range(mask_image_pil.height - value, mask_image_pil.height):
            vertical_projection_image.putpixel((x, y), 0)

    # 将 PIL 图像转换回 NumPy 数组
    horizontal_projection_image_np = np.array(horizontal_projection_image)
    vertical_projection_image_np = np.array(vertical_projection_image)

    return horizontal_projection_image_np, vertical_projection_image_np

def crop_img(src_img, br, pad=0):
    # 输入参数:
    # src_img: 原始图像(numpy array)
    # br: 裁剪矩形(x, y, w, h)，分别代表左上角坐标(x, y)以及宽度和高度
    # pad: 额外填充，默认值为0

    x, y, w, h = br
    ih, iw = src_img.shape[0:2]

    # 计算裁剪区域的边界坐标，并确保它们不超过图像范围
    y_min = np.clip(y - pad, 0, ih - 1)
    y_max = np.clip(y + h + pad, 0, ih - 1)
    x_min = np.clip(x - pad, 0, iw - 1)
    x_max = np.clip(x + w + pad, 0, iw - 1)

    # 使用numpy的切片功能对图像进行裁剪
    cropped = src_img[y_min:y_max, x_min:x_max]
    return cropped

edge_size = 10

def get_edge_pixels_rgba(image_raw):
    h, w, _ = image_raw.shape
    # 转换为RGBA格式
    image_rgba = cv2.cvtColor(image_raw, cv2.COLOR_BGR2BGRA)
    # 将非边框像素的alpha设置为0
    mask = np.ones((h, w), dtype=bool)
    mask[edge_size:-edge_size, edge_size:-edge_size] = False
    image_rgba[~mask] = [0, 0, 0, 0]
    return image_rgba

def find_dominant_color(edge_pixels_rgba):
    # 获取边框像素
    border_pixels = edge_pixels_rgba[np.where(edge_pixels_rgba[..., 3] != 0)]
    # 计算每种颜色的出现次数
    colors, counts = np.unique(border_pixels, axis=0, return_counts=True)
    # 获取出现次数最多的颜色的索引
    dominant_color_index = np.argmax(counts)
    # 计算占据边框面积的比例
    h, w, _ = edge_pixels_rgba.shape
    frame_pixels = 2 * (h + w) * edge_size - 4 * edge_size ** 2
    color_ratio = counts[dominant_color_index] / frame_pixels

    dominant_color = colors[dominant_color_index]
    return color_ratio, dominant_color