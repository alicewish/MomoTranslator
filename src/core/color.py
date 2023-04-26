from re import compile, I

from cv2 import add, inRange, mean
from loguru import logger
import numpy as np
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb, name_to_rgb, rgb_to_name

# 普通颜色与范围
def hex2int(hex_num):
    hex_num = f'0x{hex_num}'
    int_num = int(hex_num, 16)
    return int_num

def rgb2str(rgb_tuple):
    r, g, b = rgb_tuple
    color_str = f'{r:02x}{g:02x}{b:02x}'
    return color_str

def closest_color(requested_color):
    min_colors = {}
    for key, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def convert_color_to_rgb(color):
    if isinstance(color, str):
        if color.startswith("#"):
            # 十六进制颜色
            return hex_to_rgb(color)
        else:
            # 颜色名称
            return name_to_rgb(color)
    elif isinstance(color, tuple) or isinstance(color, list) or isinstance(color, np.ndarray):
        color = tuple(color)  # 如果输入是NumPy数组，将其转换为元组
        if len(color) == 3:
            # RGB颜色
            return color
        elif len(color) == 4:
            # RGBA颜色，忽略透明度
            return color[:3]
    else:
        raise ValueError("不支持的颜色格式")

def print_colored_text(text, color_str):
    # 将16进制颜色转换为RGB
    rgb_color = convert_color_to_rgb(color_str)
    r, g, b = rgb_color

    # 使用ANSI转义序列设置文本颜色
    ansi_color = f"\033[38;2;{r};{g};{b}m"
    reset_color = "\033[0m"

    # 打印彩色文本
    print(f"{ansi_color}{text}{reset_color}")

@logger.catch
def get_color_name(color):
    rgb_color = convert_color_to_rgb(color)

    try:
        color_name = rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color(rgb_color)
    return color_name


class Color:
    pattern = compile(r'([a-fA-F0-9]{6})-?(\d{0,3})', I)

    def __init__(self, color_str):
        self.type = self.__class__.__name__
        self.color_str = color_str
        self.m_color = Color.pattern.match(color_str)
        self.rgb_str = self.m_color.group(1)
        self.padding = self.m_color.group(2)

        if self.padding == '':
            self.padding = 15
        else:
            self.padding = int(self.padding)

        self.white = (self.rgb_str == 'ffffff')
        self.black = (self.rgb_str == '000000')

        self.rr = self.color_str[:2]
        self.gg = self.color_str[2:4]
        self.bb = self.color_str[4:6]

        self.r = hex2int(self.rr)
        self.g = hex2int(self.gg)
        self.b = hex2int(self.bb)
        self.a = 255
        self.rgb = (self.r, self.g, self.b)
        self.bgr = (self.b, self.g, self.r)
        self.rgba = (self.r, self.g, self.b, self.a)
        self.bgra = (self.b, self.g, self.r, self.a)

        self.ext_lower = [int(max(x - self.padding, 0)) for x in self.bgr]
        self.ext_upper = [int(min(x + self.padding, 255)) for x in self.bgr]

        self.color_name = get_color_name(self.rgb)

        if self.padding == 15:
            self.descripts = [self.rgb_str, self.color_name]
        else:
            self.descripts = [self.rgb_str, f'{self.padding}', self.color_name]
        self.descript = '-'.join(self.descripts)

    def get_img(self, task_img):

        # Bugfix: unlike jpeg format images, imdecode obtains rgba quads for png format
        image = task_img
        if task_img.shape[2] == 4:
            image = np.delete(task_img, 3, axis=2)

        frame_mask = inRange(image, np.array(self.ext_lower), np.array(self.ext_upper))
        return frame_mask

    def __str__(self):
        return f"{self.type}('{self.rgb_str}', '{self.color_name}', {self.padding})"

    def __repr__(self):
        return f'{self}'


# 渐变颜色与范围
class ColorGradient:

    def __init__(self, color_str_list):
        self.type = self.__class__.__name__
        self.color_str_list = color_str_list
        self.color_str = '-'.join(color_str_list)

        self.left_color_str, self.right_color_str = color_str_list
        self.left_color = Color(self.left_color_str)
        self.right_color = Color(self.right_color_str)

        self.rgbs = [self.left_color.rgb_str, self.right_color.rgb_str]
        self.rgb_str = '-'.join(self.rgbs)

        self.color_name = f'{self.left_color.color_name}-{self.right_color.color_name}'

        self.descripts = [self.left_color.descript, self.right_color.descript]
        self.descript = '-'.join(self.descripts)

        self.padding = int(max(self.left_color.padding, self.right_color.padding))

        self.zipped = list(zip(self.left_color.rgb, self.right_color.rgb))
        self.middle = tuple([int(mean(x)) for x in self.zipped])

        self.middle_str = rgb2str(self.middle)
        self.white = (self.middle_str == 'ffffff')
        self.black = (self.middle_str == '000000')

        self.r, self.g, self.b = self.middle
        self.a = 255
        self.rgb = (self.r, self.g, self.b)
        self.bgr = (self.b, self.g, self.r)
        self.rgba = (self.r, self.g, self.b, self.a)
        self.bgra = (self.b, self.g, self.r, self.a)

        self.ext_lower, self.ext_upper = self.get_ext(self.zipped)

    def get_ext(self, zipped, padding=15):
        lower = [min(x) for x in zipped]
        upper = [max(x) for x in zipped]

        ext_lower = [round(max(x - padding, 0)) for x in lower]
        ext_upper = [round(min(x + padding, 255)) for x in upper]

        ext_lower = tuple(reversed(ext_lower))
        ext_upper = tuple(reversed(ext_upper))
        ext = (ext_lower, ext_upper)
        return ext

    def get_img(self, task_img):

        # Bugfix: unlike jpeg format images, imdecode obtains rgba quads for png format
        image = task_img
        if task_img.shape[2] == 4:
            image = np.delete(task_img, 3, axis=2)

        frame_mask = inRange(image, np.array(self.ext_lower), np.array(self.ext_upper))
        left_sample = self.get_sample(task_img, self.left_color.rgb, self.middle)
        right_sample = self.get_sample(task_img, self.right_color.rgb, self.middle)
        img_tuple = (frame_mask, left_sample, right_sample)
        return img_tuple

    def get_sample(self, video_img, left_rgb, right_rgb):
        zipped = list(zip(left_rgb, right_rgb))
        ext_lower, ext_upper = self.get_ext(zipped)
        sample_mask = inRange(video_img, np.array(ext_lower), np.array(ext_upper))
        return sample_mask

    def __str__(self):
        return f"{self.type}('{self.rgb_str}', '{self.color_name}', {self.padding})"

    def __repr__(self):
        return f'{self}'


# 两种颜色与范围
class ColorDouble:
    def __init__(self, color_str_list):
        self.type = self.__class__.__name__
        self.color_str_list = color_str_list
        self.rgb_str = '-'.join(color_str_list)
        self.color_str = '-'.join(color_str_list)

        self.left_color_str, self.right_color_str = color_str_list
        self.left_color = Color(self.left_color_str)
        self.right_color = Color(self.right_color_str)

        self.left_color = self.left_color
        self.right_color = self.right_color

        self.rgbs = [self.left_color.rgb_str, self.right_color.rgb_str]
        self.rgb_str = '-'.join(self.rgbs)

        self.color_name = f'{self.left_color.color_name}-{self.right_color.color_name}'

        self.descripts = [self.left_color.descript, self.right_color.descript]
        self.descript = '-'.join(self.descripts)

        self.padding = int(max(self.left_color.padding, self.right_color.padding))

    def get_img(self, video_img):
        left_color_mask = self.left_color.get_img(video_img)
        right_color_mask = self.right_color.get_img(video_img)
        frame_mask = add(left_color_mask, right_color_mask)
        return frame_mask

    def __str__(self):
        return f"{self.type}('{self.rgb_str}', '{self.color_name}', {self.padding})"

    def __repr__(self):
        return f'{self}'
