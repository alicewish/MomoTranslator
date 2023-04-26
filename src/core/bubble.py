from copy import deepcopy
import cv2
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw

from scipy import ndimage as ndi
from skimage.segmentation import watershed

from storage import GlobalStorage
from helper import timer_decorator
from core.shape import Contr


# 半透明紫色 (R, G, B, A)
semi_transparent_purple = (128, 0, 128, 168)

from matplotlib import colormaps
# 使用matplotlib的tab20颜色映射
colormap_tab20 = colormaps['tab20']

def kernel(size):
    return np.ones((size, size), dtype=np.uint8)

def get_filtered_indexes(good_indexes, hierarchy):
    # 对有包含关系的轮廓，排除父轮廓
    filtered_indexes = []
    for i in range(len(good_indexes)):
        contour_index = good_indexes[i]
        parent_index = hierarchy[0][contour_index][3]
        is_parent_contour = False

        # 检查是否有任何 good_index 轮廓将此轮廓作为其父轮廓
        for other_index in good_indexes:
            hier = hierarchy[0][other_index]
            nextt, prev, first_child, parent = hier
            current_parent_index = parent

            # 检查所有上层父轮廓
            while current_parent_index != -1:
                if current_parent_index == contour_index:
                    is_parent_contour = True
                    break
                current_parent_index = hierarchy[0][current_parent_index][3]

            if is_parent_contour:
                break

        # 如果此轮廓不是任何 good_index 轮廓的父轮廓，将其添加到 filtered_indexes
        if not is_parent_contour:
            filtered_indexes.append(contour_index)
    return filtered_indexes

@timer_decorator
@logger.catch
def get_colorful_bubbles(image_raw, bubble_cnts):
    # 将原始图像转换为PIL图像
    image_pil = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
    # 创建一个与原图大小相同的透明图像
    overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = 0.5

    for f in range(len(bubble_cnts)):
        bubble_cnt = bubble_cnts[f]
        contour_points = [tuple(point[0]) for point in bubble_cnt.contour]
        # 从 tab20 颜色映射中选择一个颜色
        color = colormap_tab20(f % 20)[:3]
        color_rgba = tuple(int(c * 255) for c in color) + (int(255 * alpha),)
        draw.polygon(contour_points, fill=color_rgba)

    for f in range(len(bubble_cnts)):
        bubble_cnt = bubble_cnts[f]
        # 在轮廓中心位置添加序数
        cx, cy = bubble_cnt.cx, bubble_cnt.cy
        text = str(f + 1)
        text_bbox = draw.textbbox((cx, cy), text, font=GlobalStorage.font100, anchor="mm")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text((cx - text_width // 2, cy - text_height // 2), text, font=GlobalStorage.font100, fill=semi_transparent_purple)

    # 将带有半透明颜色轮廓的透明图像与原图混合
    image_pil.paste(overlay, mask=overlay)
    # 将 PIL 图像转换回 OpenCV 图像
    blended_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return blended_image


@timer_decorator
@logger.catch
def get_raw_bubbles(bubble_mask, letter_mask, comictextdetector_mask):
    ih, iw = bubble_mask.shape[0:2]
    black_background = np.zeros((ih, iw), dtype=np.uint8)

    all_contours, hierarchy = cv2.findContours(bubble_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    all_cnts = []
    for a in range(len(all_contours)):
        contour = all_contours[a]
        cnt = Contr(contour)
        all_cnts.append(cnt)

    good_indexes = []
    for a in range(len(all_cnts)):
        cnt = all_cnts[a]
        br_w_ratio = cnt.br_w / iw
        br_h_ratio = cnt.br_h / ih
        br_ratio = cnt.area / (cnt.br_w * cnt.br_h)
        portion_ratio = cnt.area / (iw * ih)
        # ================对轮廓数值进行初筛================
        condition_a1 = 300 <= cnt.area <= 300000
        condition_a2 = 80 <= cnt.perimeter <= 4000
        condition_a3 = 8 <= cnt.thickness <= 300
        condition_a4 = 30 <= cnt.br_w <= 3000
        condition_a5 = 30 <= cnt.br_h <= 3000
        condition_a6 = 0.01 <= br_w_ratio <= 0.9
        condition_a7 = 0.01 <= br_h_ratio <= 0.9
        condition_a8 = 0.1 <= br_ratio <= 1.1
        condition_a9 = 0.0006 <= portion_ratio <= 0.2
        condition_a10 = 0.0005 <= cnt.area_perimeter_ratio <= 0.1
        condition_as = [
            condition_a1,
            condition_a2,
            condition_a3,
            condition_a4,
            condition_a5,
            condition_a6,
            condition_a7,
            condition_a8,
            condition_a9,
            condition_a10,
        ]

        if all(condition_as):
            # ================结合文字图像进一步筛选================
            # 使用白色填充轮廓区域
            filled_contour = cv2.drawContours(black_background.copy(), [cnt.contour], 0, color=255, thickness=-1)

            # 使用位运算将 mask 和 filled_contour 相交，得到轮廓内的白色像素
            bubble_in_contour = cv2.bitwise_and(bubble_mask, filled_contour)
            letter_in_contour = cv2.bitwise_and(letter_mask, filled_contour)
            bubble_px = np.sum(bubble_in_contour == 255)
            letter_px = np.sum(letter_in_contour == 255)

            # ================对轮廓数值进行初筛================
            bubble_px_ratio = bubble_px / cnt.area  # 气泡像素占比
            letter_px_ratio = letter_px / cnt.area  # 文字像素占比
            bubble_and_letter_px_ratio = bubble_px_ratio + letter_px_ratio

            # ================气泡最外圈区域文字像素数量================
            # 步骤1：使用erode函数缩小filled_contour，得到一个缩小的轮廓
            eroded_contour = cv2.erode(filled_contour, kernel(5), iterations=1)
            # 步骤2：从原始filled_contour中减去eroded_contour，得到轮廓的边缘
            contour_edges = cv2.subtract(filled_contour, eroded_contour)
            # 步骤3：在letter_mask上统计边缘上的白色像素数量
            edge_pixels = np.where(contour_edges == 255)
            white_edge_pixel_count = 0
            for y, x in zip(*edge_pixels):
                if letter_mask[y, x] == 255:
                    white_edge_pixel_count += 1

            # ================对轮廓数值进行初筛================
            if comictextdetector_mask is not None:
                mask_in_contour = cv2.bitwise_and(comictextdetector_mask, filled_contour)
                mask_px = np.sum(mask_in_contour >= 200)
                mask_px_ratio = mask_px / cnt.area  # 文字像素占比
            else:
                mask_px_ratio = 0.1

            condition_b1 = 0.5 <= bubble_px_ratio <= 1.1
            condition_b2 = 0.001 <= letter_px_ratio <= 0.8
            condition_b3 = 0.08 <= bubble_and_letter_px_ratio <= 1.1
            condition_b4 = 0 <= white_edge_pixel_count <= 5
            condition_b5 = 0.001 <= mask_px_ratio <= 0.8
            condition_bs = [
                condition_b1,
                condition_b2,
                condition_b3,
                condition_b4,
                condition_b5,
            ]

            if all(condition_bs):
                good_indexes.append(a)

    filtered_indexes = get_filtered_indexes(good_indexes, hierarchy)

    filtered_cnts = []
    for g in range(len(filtered_indexes)):
        filtered_index = filtered_indexes[g]
        filtered_cnt = all_cnts[filtered_index]
        filtered_cnts.append(filtered_cnt)
        print(f'[{g + 1}]{filtered_cnt.br=},{filtered_cnt.area=:.0f},{filtered_cnt.perimeter=:.2f}')

    return filtered_cnts


def seg_bubbles(filtered_cnts, bubble_mask, letter_mask, media_type):
    ih, iw = bubble_mask.shape[0:2]
    black_background = np.zeros((ih, iw), dtype=np.uint8)
    zero_image = np.zeros_like(bubble_mask, dtype=np.int32)
    single_cnts = []
    for f in range(len(filtered_cnts)):
        filtered_cnt = filtered_cnts[f]
        filled_contour = cv2.drawContours(black_background.copy(), [filtered_cnt.contour], 0, color=255, thickness=-1)
        bubble_in_contour = cv2.bitwise_and(bubble_mask, filled_contour)
        letter_in_contour = cv2.bitwise_and(letter_mask, filled_contour)

        # 膨胀文字部分来估计实际气泡数
        dilate_contour = cv2.dilate(letter_in_contour, kernel(15), iterations=1)

        bulk_contours, hierarchy = cv2.findContours(dilate_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bulk_cnts = []
        for b in range(len(bulk_contours)):
            contour = bulk_contours[b]
            cnt = Contr(contour)
            bulk_cnts.append(cnt)

        good_cnts = []
        for b in range(len(bulk_cnts)):
            cnt = bulk_cnts[b]
            # ================对轮廓数值进行初筛================
            condition_a1 = 300 <= cnt.area <= 300000
            condition_a2 = 80 <= cnt.perimeter <= 4000
            condition_as = [
                condition_a1,
                condition_a2,
            ]
            if all(condition_as):
                good_cnts.append(cnt)

        if len(good_cnts) >= 2:
            if media_type != 'Manga':
                # 现在我们想要分离图像中的多个对象
                distance = ndi.distance_transform_edt(filled_contour)
                # 创建一个与图像大小相同的标记数组
                markers_full = deepcopy(zero_image)

                # 在分水岭算法中，每个标记值代表一个不同的区域，算法会将每个像素点归属到与其距离最近的标记值所代表的区域。
                for g in range(len(good_cnts)):
                    good_cnt = good_cnts[g]
                    markers_full[good_cnt.cy, good_cnt.cx] = g + 1

                markers_full = ndi.label(markers_full)[0]
                labels = watershed(-distance, markers_full, mask=filled_contour)

                # 创建一个空的彩色图像
                color_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                # 为每个标签分配一种颜色
                unique_labels = np.unique(labels)
                colors = (colormap_tab20(np.arange(len(unique_labels)))[:, :3] * 255).astype(np.uint8)

                for label, color in zip(unique_labels, colors):
                    if label == 0:
                        continue
                    color_labels[labels == label] = color

                scnts = []
                for u in range(len(unique_labels)):
                    target_label = unique_labels[u]
                    binary_image = np.zeros_like(labels, dtype=np.uint8)
                    binary_image[labels == target_label] = 255

                    # 查找轮廓
                    sep_contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for s in range(len(sep_contours)):
                        contour = sep_contours[s]
                        cnt = Contr(contour)
                        # ================对轮廓数值进行初筛================
                        condition_a1 = 300 <= cnt.area <= 300000
                        condition_a2 = 80 <= cnt.perimeter <= 4000
                        condition_as = [
                            condition_a1,
                            condition_a2,
                        ]
                        if all(condition_as):
                            scnts.append(cnt)
                if len(scnts) == len(good_cnts):
                    single_cnts.extend(scnts)
                else:
                    single_cnts.append(filtered_cnt)
                    # TODO分水岭切割失败
            else:
                single_cnts.append(filtered_cnt)
        else:
            single_cnts.append(filtered_cnt)

    return single_cnts