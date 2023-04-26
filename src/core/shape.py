import cv2
import numpy as np

edge_ratio = 0.135

# 框架宽度
grid_width_min = 100
# 框架高度
grid_height_min = 100

grid_ratio_dic = {
    'Manga': 0.95,  # 日漫黑白
    'Comic': 0.95,  # 美漫彩色
}

class Contr:

    def __init__(self, contour):
        self.type = self.__class__.__name__
        self.contour = contour

        self.area = cv2.contourArea(self.contour)
        self.perimeter = cv2.arcLength(self.contour, True)
        if self.perimeter != 0:
            self.thickness = self.area / self.perimeter
            self.area_perimeter_ratio = self.area / self.perimeter / self.perimeter
        else:
            self.thickness = 0
            self.area_perimeter_ratio = 0

        self.br = cv2.boundingRect(self.contour)
        self.br_x, self.br_y, self.br_w, self.br_h = self.br
        self.br_u = self.br_x + self.br_w
        self.br_v = self.br_y + self.br_h
        self.br_m = int(self.br_x + 0.5 * self.br_w)
        self.br_n = int(self.br_y + 0.5 * self.br_h)
        self.br_area = self.br_w * self.br_h
        self.br_rt = self.area / self.br_area

        self.M = cv2.moments(self.contour)
        # 这两行是计算质点坐标
        if self.M['m00'] != 0:
            self.cx = int(self.M['m10'] / self.M['m00'])
            self.cy = int(self.M['m01'] / self.M['m00'])
        else:
            self.cx, self.cy = 0, 0
        self.dd = (self.cx, self.cy)
        self.ee = (self.cy, self.cx)

    def add_cropped_image(self, cropped_image):
        self.cropped_image = cropped_image

    def __str__(self):
        return f"{self.type}({self.area:.1f}, {self.perimeter:.2f})"

    def __repr__(self):
        return f'{self}'


class Rect:
    descript = ''

    def __init__(self, rect_tup, media_type):
        self.type = self.__class__.__name__
        self.rect_tup = rect_tup
        self.media_type = media_type

        self.x, self.y, self.w, self.h = self.rect_tup
        self.rect_inner_tuple = self.rect_tup

        self.x_max = self.x + self.w
        self.y_max = self.y + self.h

        self.area = self.w * self.h
        self.basic = False

    def get_devide(self, frame_mask):
        self.sub_frame_mask = frame_mask[self.y:self.y_max, self.x:self.x_max]
        self.pg = PicGrid(
            self.sub_frame_mask,
            self.media_type,
            self.x,
            self.y,
        )
        self.a, self.b, self.c, self.d = 0, 0, 0, 0

        # 水平线
        if self.pg.white_lines_horizontal:
            self.dep = self.pg.ih

            first_white_line = self.pg.white_lines_horizontal[0]
            if first_white_line[0] <= edge_ratio * self.dep:
                self.a = first_white_line[0] + first_white_line[1]

            last_white_line = self.pg.white_lines_horizontal[-1]
            if last_white_line[0] + last_white_line[1] >= (1 - edge_ratio) * self.dep:
                self.b = self.dep - last_white_line[0]

        # 竖直线
        if self.pg.white_lines_vertical:
            self.dep = self.pg.iw

            first_white_line = self.pg.white_lines_vertical[0]
            if first_white_line[0] <= edge_ratio * self.dep:
                self.c = first_white_line[0] + first_white_line[1]

            last_white_line = self.pg.white_lines_vertical[-1]
            if last_white_line[0] + last_white_line[1] >= (1 - edge_ratio) * self.dep:
                self.d = self.dep - last_white_line[0]

        self.xx = self.x + self.c
        self.yy = self.y + self.a
        self.ww = self.w - self.c - self.d
        self.hh = self.h - self.a - self.b

        self.rect_inner_tuple = (self.xx, self.yy, self.ww, self.hh)
        self.frame_grid = [self.rect_tup, self.rect_inner_tuple]
        self.frame_grid_str = '~'.join([','.join([f'{y}' for y in x]) for x in self.frame_grid])

    def __str__(self):
        return f"{self.type}({self.x}, {self.y}, {self.w}, {self.h})"

    def __repr__(self):
        return f'{self}'

def get_segment(binary_mask, method, media_type):
    grid_ratio = grid_ratio_dic.get(media_type, 0.95)
    ih, iw = binary_mask.shape[0:2]

    if method == 'horizontal':  # 水平线
        dep = ih
        leng = iw
        picture = binary_mask
    else:  # method=='vertical' # 竖直线
        dep = iw
        leng = ih
        picture = binary_mask.T

    # ================粗筛================
    # 框架白线最短长度
    min_leng = grid_ratio * leng

    # 投影
    columns = (picture != 0).sum(0)
    rows = (picture != 0).sum(1)

    # 找出白线
    ori_gray = np.where(rows >= min_leng)[0]

    # 对相邻白线分组
    cons = np.split(ori_gray, np.where(np.diff(ori_gray) != 1)[0] + 1)

    # 转化为投影线段
    white_lines = [(x[0], len(x) - 1) for x in cons if len(x) >= 1]

    # ================最小厚度检验================
    white_lines_correct = [x for x in white_lines if x[1] > 1]
    return white_lines_correct


def get_recs(method, white_lines, iw, ih, offset_x, offset_y, media_type):
    y_min, y_max, x_min, x_max = 0, 0, 0, 0
    main_rec = (0, 0, iw, ih)

    if method == 'horizontal':
        x_min = 0
        x_max = iw
        dep = ih
    else:  # method=='vertical'
        y_min = 0
        y_max = ih
        dep = iw

    true_dividers = [0]

    for pind in range(len(white_lines)):
        tu = white_lines[pind]
        start_point, leng = tu
        # divider = round(start_point + 0.5 * leng)
        if edge_ratio * dep <= start_point < start_point + leng <= (1 - edge_ratio) * dep:
            true_dividers.append(start_point)

    true_dividers.append(dep)
    rectangles = []
    valid = True

    small_box_num = 0
    for pind in range(len(true_dividers) - 1):
        upper = true_dividers[pind]
        lower = true_dividers[pind + 1]

        if method == 'horizontal':
            y_min = upper
            y_max = lower
        else:  # method=='vertical'
            x_min = upper
            x_max = lower

        w = x_max - x_min
        h = y_max - y_min

        x = x_min + offset_x
        y = y_min + offset_y

        rectangle = (x, y, w, h)

        # 如果切分时有矩形太小则不放入划分列表
        if w <= grid_width_min or h <= grid_height_min:
            print(f'{rectangle=}')
            small_box_num += 1
        else:
            rectangles.append(rectangle)

    # 如果切分时有多个矩形太小则不合理
    if small_box_num >= 2:
        valid = False

    if not valid:
        rectangles = [main_rec]

    # 日漫从右到左，其他从左到右
    if media_type == 'Manga' and method == 'vertical':
        rectangles.reverse()

    return rectangles

class PicGrid:

    def __init__(self, binary_mask, media_type, offset_x=0, offset_y=0):
        self.type = self.__class__.__name__
        self.frame_mask = binary_mask
        self.media_type = media_type
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.basic = False
        self.judgement = None

        # ================长宽================
        self.ih, self.iw = self.frame_mask.shape[0:2]
        self.px_pts = np.transpose(np.nonzero(binary_mask))
        self.px_area = self.px_pts.shape[0]

        self.white_lines_horizontal = get_segment(self.frame_mask, 'horizontal', media_type)
        self.white_lines_vertical = get_segment(self.frame_mask, 'vertical', media_type)
        self.white_lines = {
            'horizontal': self.white_lines_horizontal,
            'vertical': self.white_lines_vertical}

        self.rects_horizontal = get_recs(
            'horizontal',
            self.white_lines_horizontal,
            self.iw,
            self.ih,
            self.offset_x,
            self.offset_y,
            self.media_type,
        )
        self.rects_vertical = get_recs(
            'vertical',
            self.white_lines_vertical,
            self.iw,
            self.ih,
            self.offset_x,
            self.offset_y,
            self.media_type,
        )
        self.rects_dic = {
            'horizontal': self.rects_horizontal,
            'vertical': self.rects_vertical}

        self.rects = None
        if len(self.rects_horizontal) == len(self.rects_vertical) == 1:
            self.basic = True
        elif len(self.rects_horizontal) >= len(self.rects_vertical):
            # 矩形排列方向
            self.judgement = 'horizontal'
        else:
            self.judgement = 'vertical'
        if self.judgement:
            self.rects = self.rects_dic[self.judgement]

    def __str__(self):
        return f"{self.type}({self.iw}, {self.ih}, {self.px_area}, '{self.judgement}')"

    def __repr__(self):
        return f'{self}'