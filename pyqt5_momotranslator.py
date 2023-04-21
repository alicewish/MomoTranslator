import codecs
import io
import os
import os.path
import re
import sys
import webbrowser
from ast import Import, ImportFrom, parse, walk
from base64 import b64encode
from copy import deepcopy
from csv import reader, writer
from functools import wraps
from getpass import getuser
from hashlib import md5
from itertools import chain
from locale import getdefaultlocale
from operator import mod
from os.path import abspath, dirname, exists, expanduser, getsize, isfile
from pathlib import Path
from platform import system, uname
from pprint import pprint
from re import I, findall
from re import compile as recompile
from shutil import copy2
from stdlib_list import stdlib_list
from subprocess import Popen, call
from time import strftime, time
from traceback import print_exc
from uuid import getnode

import numpy as np
import pkg_resources
import xmltodict
import yaml
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QSettings, QSize, QTranslator, Qt
from PyQt6.QtGui import QAction, QActionGroup, QBrush, QDoubleValidator, QFont, QGuiApplication, QIcon, QImage, \
    QKeySequence, QPainter, QPixmap, QTransform
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QDockWidget, QFileDialog, \
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, \
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QRadioButton, QStatusBar, QTabWidget, \
    QToolBar, QToolButton, QVBoxLayout, QWidget
from aip import AipOcr
from cv2 import BORDER_CONSTANT, CHAIN_APPROX_SIMPLE, COLOR_BGR2BGRA, COLOR_BGR2RGB, COLOR_RGB2BGR, INTER_LINEAR, \
    RETR_EXTERNAL, RETR_TREE, add, arcLength, bitwise_and, boundingRect, contourArea, copyMakeBorder, cvtColor, dilate, \
    dnn, drawContours, erode, findContours, imdecode, imencode, inRange, mean, moments, resize, subtract
from deep_translator import GoogleTranslator
from docx import Document
from docx.shared import Inches
from easyocr import Reader
from loguru import logger
from matplotlib import colormaps
from natsort import natsorted
from numpy import arange, argmax, array, ascontiguousarray, clip, diff, float32, fromfile, int32, maximum, mean, \
    minimum, mod, ndarray, nonzero, ones, transpose, uint8, unique, where, zeros, zeros_like
from numpy import split as n_split
from pytesseract import image_to_string
from qtawesome import icon
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb, name_to_rgb, rgb_to_name

use_torch = True
use_torch = False
if use_torch:
    import torch


# ================================参数区================================

def a1_const():
    return


# Platforms
SYSTEM = ''
platform_system = system()
platform_uname = uname()
os_kernal = platform_uname.machine
if os_kernal in ['x86_64', 'AMD64']:
    if platform_system == 'Windows':
        SYSTEM = 'WINDOWS'
    elif platform_system == 'Linux':
        SYSTEM = 'LINUX'
    else:  # 'Darwin'
        SYSTEM = 'MAC'
else:  # os_kernal = 'arm64'
    if platform_system == 'Windows':
        SYSTEM = 'WINDOWS'
    elif platform_system == 'Darwin':
        SYSTEM = 'M1'
    else:
        SYSTEM = 'PI'

locale_tup = getdefaultlocale()
lang_code = locale_tup[0]

username = getuser()
homedir = expanduser("~")
DOWNLOADS = Path(homedir) / 'Downloads'
DOCUMENTS = Path(homedir) / 'Documents'

mac_address = ':'.join(findall('..', '%012x' % getnode()))
node_name = platform_uname.node

current_dir = dirname(abspath(__file__))
current_dir = Path(current_dir)

dirpath = os.getcwd()
ProgramFolder = Path(dirpath)
UserDataFolder = ProgramFolder / 'MomoHanhuaUserData'

if SYSTEM == 'WINDOWS':
    encoding = 'gbk'
    line_feed = '\n'
    cmct = 'ctrl'
else:
    encoding = 'utf-8'
    line_feed = '\n'
    cmct = 'command'

if SYSTEM in ['MAC', 'M1']:
    from Quartz import kCGRenderingIntentDefault
    from Quartz.CoreGraphics import CGDataProviderCreateWithData, CGColorSpaceCreateDeviceRGB, CGImageCreate
    from Vision import VNImageRequestHandler, VNRecognizeTextRequest

line_feeds = line_feed * 2

lf = line_feed
lfs = line_feeds

ignores = ('~$', '._')

type_dic = {
    'xlsx': '.xlsx',
    'csv': '.csv',
    'pr': '.prproj',
    'psd': '.psd',
    'tor': '.torrent',
    'xml': '.xml',
    'audio': ('.aif', '.mp3', '.wav', '.flac', '.m4a', '.ogg'),
    'video': ('.mp4', '.mkv', '.avi', '.flv', '.mov', '.wmv'),
    'compressed': ('.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'),
    'font': ('.ttc', '.ttf', '.otf'),
    'comic': ('.cbr', '.cbz', '.rar', '.zip', '.pdf', '.txt'),
    'pic': ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'),
    'log': '.log',
    'json': '.json',
    'pickle': '.pkl',
    'python': '.py',
    'txt': '.txt',
    'doc': ('.doc', '.docx'),
    'ppt': ('.ppt', '.pptx'),
    'pdf': '.pdf',
    'html': ('.html', '.htm'),
    'css': '.css',
    'js': '.js',
    'markdown': ('.md', '.markdown'),
}

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

APP_NAME = 'MomoTranslator'
MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0
APP_VERSION = f'v{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}'

APP_AUTHOR = '墨问非名'

video_width = 1920
video_height = 1080
video_size = (video_width, video_height)

pos_0 = (0, 0)
pos_c = ('center', 'center')
pylupdate = 'pylupdate6'
lrelease = 'lrelease'

window_title_prefix = f'{APP_NAME} {APP_VERSION}'

py_path = Path(__file__).resolve()
py_dev_path = py_path.parent / f'{py_path.stem}_dev.py'

max_chars = 5000

pictures_exclude = '加框,分框,框,涂白,填字,修图,-,copy,副本,拷贝,顺序,打码,测试,标注,边缘,标志,伪造'
pic_tuple = tuple(pictures_exclude.split(','))

image_format = 'jpeg'
# image_format = 'png'

media_type_dict = {
    0: 'Comic',
    1: 'Manga',
}

media_language_dict = {
    0: 'English',
    1: 'Chinese Simplified',
    2: 'Chinese Traditional',
    3: 'Japanese',
    4: 'Korean',
}

special_keywords = [
    'setIcon', 'icon',
    'setShortcut', 'QKeySequence',
    'QSettings', 'value', 'setValue',
    'triggered',
    'setWindowTitle', 'windowTitle',
]

language_tuples = [
    # ================支持的语言================
    ('zh_CN', 'Simplified Chinese', '简体中文', '简体中文'),
    ('zh_TW', 'Traditional Chinese', '繁体中文', '繁體中文'),
    ('en_US', 'English', '英语', 'English'),
    ('ja_JP', 'Japanese', '日语', '日本語'),
    ('ko_KR', 'Korean', '韩语', '한국어'),
    # ================未来支持的语言================
    # ('es_ES', 'Spanish', '西班牙语', 'Español'),
    # ('fr_FR', 'French', '法语', 'Français'),
    # ('de_DE', 'German', '德语', 'Deutsch'),
    # ('it_IT', 'Italian', '意大利语', 'Italiano'),
    # ('pt_PT', 'Portuguese', '葡萄牙语', 'Português'),
    # ('ru_RU', 'Russian', '俄语', 'Русский'),
    # ('ar_AR', 'Arabic', '阿拉伯语', 'العربية'),
    # ('nl_NL', 'Dutch', '荷兰语', 'Nederlands'),
    # ('sv_SE', 'Swedish', '瑞典语', 'Svenska'),
    # ('tr_TR', 'Turkish', '土耳其语', 'Türkçe'),
    # ('pl_PL', 'Polish', '波兰语', 'Polski'),
    # ('he_IL', 'Hebrew', '希伯来语', 'עברית'),
    # ('da_DK', 'Danish', '丹麦语', 'Dansk'),
    # ('fi_FI', 'Finnish', '芬兰语', 'Suomi'),
    # ('no_NO', 'Norwegian', '挪威语', 'Norsk'),
    # ('hu_HU', 'Hungarian', '匈牙利语', 'Magyar'),
    # ('cs_CZ', 'Czech', '捷克语', 'Čeština'),
    # ('ro_RO', 'Romanian', '罗马尼亚语', 'Română'),
    # ('el_GR', 'Greek', '希腊语', 'Ελληνικά'),
    # ('id_ID', 'Indonesian', '印度尼西亚语', 'Bahasa Indonesia'),
    # ('th_TH', 'Thai', '泰语', 'ภาษาไทย'),
]

input_size = 1024
input_tuple = (input_size, input_size)
device = 'cpu'
half = False
to_tensor = False
auto = False
scaleFill = False
scaleup = True
stride = 64
thresh = None

edge_size = 10
grid_ratio_dic = {
    'Manga': 0.95,  # 日漫黑白
    'Comic': 0.95,  # 美漫彩色
}
edge_ratio = 0.135
# 框架宽度
grid_width_min = 100
# 框架高度
grid_height_min = 100

color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_white = (255, 255, 255)
color_black = (0, 0, 0)

rgba_white = (255, 255, 255, 255)
rgba_zero = (0, 0, 0, 0)
rgba_black = (0, 0, 0, 255)

# 半透明紫色 (R, G, B, A)
semi_transparent_purple = (128, 0, 128, 168)

padding = 10

p_color = recompile(r'([a-fA-F0-9]{6})-?(\d{0,3})', I)

# 使用matplotlib的tab20颜色映射
colormap_tab20 = colormaps['tab20']

font_path = "Arial.Unicode.ttf"
font60 = ImageFont.truetype(font_path, 60)
font100 = ImageFont.truetype(font_path, 100)

tesseract_language_options = {
    'Chinese Simplified': 'chi_sim',
    'Chinese Traditional': 'chi_tra',
    'English': 'eng',
    'Japanese': 'jpn',
    'Korean': 'kor',
}

vision_language_options = {
    'Chinese Simplified': 'zh-Hans',
    'Chinese Traditional': 'zh-Hant',
    'English': 'en',
    'Japanese': 'ja',
    'Korean': 'ko',
}

baidu_language_options = {
    'Chinese Simplified': 'CHN_ENG',
    'Chinese Traditional': 'CHN_ENG',
    'English': 'ENG',
    'Japanese': 'JPN',
    'Korean': 'KOR',
}

# 请使用您自己的百度OCR应用密钥
BAIDU_APP_ID = 'your_baidu_app_id'
BAIDU_API_KEY = 'your_baidu_api_key'
BAIDU_SECRET_KEY = 'your_baidu_secret_key'


def kernel(size):
    return ones((size, size), uint8)


kernel3 = kernel(3)
kernel5 = kernel(5)
kernel10 = kernel(10)
kernel12 = kernel(12)
kernel15 = kernel(15)
kernel20 = kernel(20)


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        elapsed_time = time() - start_time

        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        if hours > 0:
            show_run_time = f"{int(hours)}时{int(minutes)}分{seconds:.2f}秒"
        elif minutes > 0:
            show_run_time = f"{int(minutes)}分{seconds:.2f}秒"
        else:
            show_run_time = f"{seconds:.2f}秒"

        logger.debug(f"{func.__name__} took: {show_run_time}")
        return result

    return wrapper


def is_decimal_or_comma(s):
    pattern = r'^\d*\.?\d*$|^\d*[,]?\d*$'
    return bool(re.match(pattern, s))


def is_valid_file(file_path, suffixes):
    if not file_path.is_file():
        return False
    if not file_path.stem.startswith(ignores):
        if suffixes:
            return file_path.suffix.lower() in suffixes
        else:
            return True
    return False


def printe(e):
    print(e)
    logger.error(e)
    print_exc()


# ================创建目录================

def make_dir(file_path):
    if not exists(file_path):
        try:
            os.makedirs(file_path)
        except BaseException as e:
            print(e)


# ================获取文件夹列表================

def get_dirs(rootdir):
    dirs_list = []
    if rootdir and rootdir.exists():
        # 列出目录下的所有文件和目录
        lines = os.listdir(rootdir)
        for line in lines:
            filepath = Path(rootdir) / line
            if filepath.is_dir():
                dirs_list.append(filepath)
    dirs_list.sort()
    return dirs_list


def get_files(rootdir, file_type=None, direct=False):
    rootdir = Path(rootdir)
    file_paths = []

    # 获取文件类型的后缀
    # 默认为所有文件
    suffixes = type_dic.get(file_type, file_type)
    if isinstance(suffixes, str):
        suffixes = (suffixes,)

    # 如果根目录存在
    if rootdir and rootdir.exists():
        # 只读取当前文件夹下的文件
        if direct:
            files = os.listdir(rootdir)
            for file in files:
                file_path = Path(rootdir) / file
                if is_valid_file(file_path, suffixes):
                    file_paths.append(file_path)
        # 读取所有文件
        else:
            for root, dirs, files in os.walk(rootdir):
                for file in files:
                    file_path = Path(root) / file
                    if is_valid_file(file_path, suffixes):
                        file_paths.append(file_path)

    # 使用natsorted()进行自然排序，
    # 使列表中的字符串按照数字顺序进行排序
    file_paths = natsorted(file_paths)
    return file_paths


# @logger.catch
def desel_list(old_list, tup, method='stem'):
    if method == 'name':
        new_list = [x for x in old_list if not x.name.endswith(tup)]
    else:
        new_list = [x for x in old_list if not x.stem.endswith(tup)]
    return new_list


def get_valid_image_list(rootdir, mode='raw'):
    all_pics = get_files(rootdir, 'pic', True)
    jpgs = [x for x in all_pics if x.suffix == '.jpg']
    pngs = [x for x in all_pics if x.suffix == '.png']

    all_masks = [x for x in pngs if '-Mask-' in x.stem]
    no_masks = [x for x in pngs if '-Mask-' not in x.stem]

    valid_jpgs = desel_list(jpgs, pic_tuple)
    valid_pngs = desel_list(no_masks, pic_tuple)

    valid_image_list = []
    if valid_jpgs:
        valid_image_list = valid_jpgs
    elif valid_pngs:
        valid_image_list = valid_pngs
    if mode == 'raw':
        return valid_image_list
    else:
        return all_masks


# ================读取文本================

def read_txt(file_path, encoding='utf-8'):
    """
    读取指定文件路径的文本内容。

    :param file_path: 文件路径
    :param encoding: 文件编码，默认为'utf-8'
    :return: 返回读取到的文件内容，如果文件不存在则返回None
    """
    file_content = None
    if file_path.exists():
        with open(file_path, mode='r', encoding=encoding) as file_object:
            file_content = file_object.read()
    return file_content


# ================写入文件================

def write_txt(file_path, text_input, encoding='utf-8', ignore_empty=True):
    """
    将文本内容写入指定的文件路径。

    :param file_path: 文件路径
    :param text_input: 要写入的文本内容，可以是字符串或字符串列表
    :param encoding: 文件编码，默认为'utf-8'
    :param ignore_empty: 是否忽略空内容，默认为True
    """
    if text_input:
        save_text = True

        if isinstance(text_input, list):
            otext = lf.join(text_input)
        else:
            otext = text_input

        file_content = read_txt(file_path, encoding)

        if file_content == otext or (ignore_empty and otext == ''):
            save_text = False

        if save_text:
            with open(file_path, mode='w', encoding=encoding, errors='ignore') as f:
                f.write(otext)


# ================对文件算MD5================

def md5_w_size(path, blksize=2 ** 20):
    if isfile(path) and exists(path):  # 判断目标是否文件,及是否存在
        file_size = getsize(path)
        if file_size <= 256 * 1024 * 1024:  # 512MB
            with open(path, 'rb') as f:
                cont = f.read()
            hash_object = md5(cont)
            t_md5 = hash_object.hexdigest()
            return t_md5, file_size
        else:
            m = md5()
            with open(path, 'rb') as f:
                while True:
                    buf = f.read(blksize)
                    if not buf:
                        break
                    m.update(buf)
            t_md5 = m.hexdigest()
            return t_md5, file_size
    else:
        return None


def write_csv(csv_path, data_input, headers=None):
    temp_csv = csv_path.parent / 'temp.csv'

    try:
        if isinstance(data_input, list):
            if len(data_input) >= 1:
                if csv_path.exists():
                    with codecs.open(temp_csv, 'w', 'utf_8_sig') as f:
                        f_csv = writer(f)
                        if headers:
                            f_csv.writerow(headers)
                        f_csv.writerows(data_input)
                    if md5_w_size(temp_csv) != md5_w_size(csv_path):
                        copy2(temp_csv, csv_path)
                    if temp_csv.exists():
                        os.remove(temp_csv)
                else:
                    with codecs.open(csv_path, 'w', 'utf_8_sig') as f:
                        f_csv = writer(f)
                        if headers:
                            f_csv.writerow(headers)
                        f_csv.writerows(data_input)
        else:  # DataFrame
            if csv_path.exists():
                data_input.to_csv(temp_csv, encoding='utf-8', index=False)
                if md5_w_size(temp_csv) != md5_w_size(csv_path):
                    copy2(temp_csv, csv_path)
                if temp_csv.exists():
                    os.remove(temp_csv)
            else:
                data_input.to_csv(csv_path, encoding='utf-8', index=False)
    except BaseException as e:
        printe(e)


def write_pic(pic_path, picimg):
    pic_path = Path(pic_path)
    ext = pic_path.suffix
    temp_pic = pic_path.parent / f'temp{ext}'

    # 检查输入图像的类型，如果是PIL图像，则将其转换为NumPy数组
    if isinstance(picimg, Image.Image):
        picimg = array(picimg)

    # 保存临时图像
    imencode(ext, picimg)[1].tofile(temp_pic)

    # 检查临时图像和目标图像的md5哈希和大小是否相同
    if not pic_path.exists() or md5_w_size(temp_pic) != md5_w_size(pic_path):
        copy2(temp_pic, pic_path)

    # 删除临时图像
    if temp_pic.exists():
        os.remove(temp_pic)

    return pic_path


def get_edge_pixels_rgba(image_raw):
    h, w, _ = image_raw.shape
    # 转换为RGBA格式
    image_rgba = cvtColor(image_raw, COLOR_BGR2BGRA)
    # 将非边框像素的alpha设置为0
    mask = ones((h, w), dtype=bool)
    mask[edge_size:-edge_size, edge_size:-edge_size] = False
    image_rgba[~mask] = [0, 0, 0, 0]
    return image_rgba


def find_dominant_color(edge_pixels_rgba):
    # 获取边框像素
    border_pixels = edge_pixels_rgba[where(edge_pixels_rgba[..., 3] != 0)]
    # 计算每种颜色的出现次数
    colors, counts = unique(border_pixels, axis=0, return_counts=True)
    # 获取出现次数最多的颜色的索引
    dominant_color_index = argmax(counts)
    # 计算占据边框面积的比例
    h, w, _ = edge_pixels_rgba.shape
    frame_pixels = 2 * (h + w) * edge_size - 4 * edge_size ** 2
    color_ratio = counts[dominant_color_index] / frame_pixels

    dominant_color = colors[dominant_color_index]
    return color_ratio, dominant_color


def convert_color_to_rgb(color):
    if isinstance(color, str):
        if color.startswith("#"):
            # 十六进制颜色
            return hex_to_rgb(color)
        else:
            # 颜色名称
            return name_to_rgb(color)
    elif isinstance(color, tuple) or isinstance(color, list) or isinstance(color, ndarray):
        color = tuple(color)  # 如果输入是NumPy数组，将其转换为元组
        if len(color) == 3:
            # RGB颜色
            return color
        elif len(color) == 4:
            # RGBA颜色，忽略透明度
            return color[:3]
    else:
        raise ValueError("不支持的颜色格式")


def closest_color(requested_color):
    min_colors = {}
    for key, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


@logger.catch
def get_color_name(color):
    rgb_color = convert_color_to_rgb(color)

    try:
        color_name = rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color(rgb_color)
    return color_name


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
def get_marked_frames(frame_grid_strs, image_raw):
    image_rgb = cvtColor(image_raw, COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    for f in range(len(frame_grid_strs)):
        frame_grid_str = frame_grid_strs[f]
        # 使用正则表达式提取字符串中的所有整数
        int_values = list(map(int, findall(r'\d+', frame_grid_str)))
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
        draw.text(index_position, str(f + 1), fill=(0, 0, 255, 255), font=font60)

    marked_frames = cvtColor(array(pil_image), COLOR_RGB2BGR)
    return marked_frames


def crop_img(src_img, br, pad=0):
    # 输入参数:
    # src_img: 原始图像(numpy array)
    # br: 裁剪矩形(x, y, w, h)，分别代表左上角坐标(x, y)以及宽度和高度
    # pad: 额外填充，默认值为0

    x, y, w, h = br
    ih, iw = src_img.shape[0:2]

    # 计算裁剪区域的边界坐标，并确保它们不超过图像范围
    y_min = clip(y - pad, 0, ih - 1)
    y_max = clip(y + h + pad, 0, ih - 1)
    x_min = clip(x - pad, 0, iw - 1)
    x_max = clip(x + w + pad, 0, iw - 1)

    # 使用numpy的切片功能对图像进行裁剪
    cropped = src_img[y_min:y_max, x_min:x_max]
    return cropped


def hex2int(hex_num):
    hex_num = f'0x{hex_num}'
    int_num = int(hex_num, 16)
    return int_num


def rgb2str(rgb_tuple):
    r, g, b = rgb_tuple
    color_str = f'{r:02x}{g:02x}{b:02x}'
    return color_str


def wrap_strings_with_tr():
    content = read_txt(py_path)

    # 匹配从 QWidget 或 QMainWindow 继承的类
    class_pattern = r'class\s+\w+\s*\((?:\s*\w+\s*,\s*)*(QWidget|QMainWindow)(?:\s*,\s*\w+)*\s*\)\s*:'
    class_matches = re.finditer(class_pattern, content)

    # 在这些类中查找没有包裹在 self.tr() 中的字符串
    string_pattern = r'(?<!self\.tr\()\s*(".*?"|\'.*?\')(?!\s*\.(?:format|replace)|\s*\%|settings\.value)(?!\s*(?:setIcon|icon|setShortcut|QKeySequence|QSettings)\s*\()'

    for class_match in class_matches:
        class_start = class_match.start()
        next_match = re.search(r'class\s+\w+\s*\(', content[class_start + 1:])
        if next_match:
            class_end = next_match.start()
        else:
            class_end = len(content)

        class_content = content[class_start:class_end]

        matches = re.finditer(string_pattern, class_content)
        new_class_content = class_content
        for match in reversed(list(matches)):
            # 检查是否需要替换文本
            str_value = match.group(1)[1:-1]

            # 获取字符串所在的整行
            line_start = class_content.rfind('\n', 0, match.start()) + 1
            line_end = class_content.find('\n', match.end())
            line = class_content[line_start:line_end]

            if is_decimal_or_comma(str_value) or str_value.endswith('%'):
                continue

            # 检查行中是否包含我们不需要替换的关键字
            if any(keyword in line for keyword in special_keywords):
                continue

            start = match.start(1)
            end = match.end(1)
            logger.info(f'正在修改: {match.group(1)}')
            # 使用单引号包裹字符串
            new_class_content = new_class_content[
                                :start] + f'self.tr(\'{match.group(1)[1:-1]}\')' + new_class_content[end:]
        content = content[:class_start] + new_class_content + content[class_start + class_end:]
    updated_content = content

    print(updated_content)
    write_txt(py_dev_path, updated_content)


def iread_csv(csv_file, pop_head=True, get_head=False):
    with open(csv_file, encoding='utf-8', mode='r') as f:
        f_csv = reader(f)
        if pop_head:
            head = next(f_csv, [])  # 获取首行并在需要时将其从数据中删除
        else:
            head = []
        idata = [tuple(row) for row in f_csv]  # 使用列表推导式简化数据读取
    if get_head:
        return idata, head
    else:
        return idata


def compute_frame_mask(image_raw, dominant_color, tolerance=5):
    dominant_color_int32 = dominant_color[:3].astype(int32)
    lower_bound = maximum(0, dominant_color_int32 - tolerance)
    upper_bound = minimum(255, dominant_color_int32 + tolerance)
    mask = inRange(image_raw, lower_bound, upper_bound)
    return mask


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
    horizontal_projection_image_np = array(horizontal_projection_image)
    vertical_projection_image_np = array(vertical_projection_image)

    return horizontal_projection_image_np, vertical_projection_image_np


def find_intervals(projection, threshold):
    intervals = []  # 用于存储符合条件的区间
    start = None  # 区间的起始点
    length = 0  # 区间的长度

    # 遍历投影数据
    for i, value in enumerate(projection):
        # 如果当前值大于等于阈值
        if value >= threshold:
            # 如果区间尚未开始，设置起始点为当前位置
            if start is None:
                start = i
            # 区间长度加1
            length += 1
        else:
            # 如果区间已经开始，将区间添加到结果列表中
            if start is not None:
                intervals.append((start, length))
                start = None
                length = 0

    # 如果循环结束时区间仍然有效，将区间添加到结果列表中
    if start is not None:
        intervals.append((start, length))

    return intervals


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
    ori_gray = where(rows >= min_leng)[0]

    # 对相邻白线分组
    cons = n_split(ori_gray, where(diff(ori_gray) != 1)[0] + 1)

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


# 普通颜色与范围
class Color:

    def __init__(self, color_str):
        self.type = self.__class__.__name__
        self.color_str = color_str
        self.m_color = p_color.match(color_str)
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
        frame_mask = inRange(task_img, array(self.ext_lower), array(self.ext_upper))
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
        frame_mask = inRange(task_img, array(self.ext_lower), array(self.ext_upper))
        left_sample = self.get_sample(task_img, self.left_color.rgb, self.middle)
        right_sample = self.get_sample(task_img, self.right_color.rgb, self.middle)
        img_tuple = (frame_mask, left_sample, right_sample)
        return img_tuple

    def get_sample(self, video_img, left_rgb, right_rgb):
        zipped = list(zip(left_rgb, right_rgb))
        ext_lower, ext_upper = self.get_ext(zipped)
        sample_mask = inRange(video_img, array(ext_lower), array(ext_upper))
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


class Contr:

    def __init__(self, contour):
        self.type = self.__class__.__name__
        self.contour = contour

        self.area = contourArea(self.contour)
        self.perimeter = arcLength(self.contour, True)
        if self.perimeter != 0:
            self.thickness = self.area / self.perimeter
            self.area_perimeter_ratio = self.area / self.perimeter / self.perimeter
        else:
            self.thickness = 0
            self.area_perimeter_ratio = 0

        self.br = boundingRect(self.contour)
        self.br_x, self.br_y, self.br_w, self.br_h = self.br
        self.br_u = self.br_x + self.br_w
        self.br_v = self.br_y + self.br_h
        self.br_m = int(self.br_x + 0.5 * self.br_w)
        self.br_n = int(self.br_y + 0.5 * self.br_h)
        self.br_area = self.br_w * self.br_h
        self.br_rt = self.area / self.br_area

        self.M = moments(self.contour)
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
        self.px_pts = transpose(nonzero(binary_mask))
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


def get_frame_grid_recursive(rect, grids, frame_mask, media_type):
    rec = Rect(rect, media_type)
    rec.get_devide(frame_mask)

    if rec.pg.basic:
        grids.append(rec)
    else:
        rects = rec.pg.rects
        for r in rects:
            get_frame_grid_recursive(r, grids, frame_mask, media_type)


def ocr_by_tesseract(image, media_language, vertical=False):
    # 获取所选语言的配置
    language_config = tesseract_language_options[media_language]
    config = f'-c preserve_interword_spaces=1 --psm 6 -l {language_config}'
    # 如果识别竖排文本
    if vertical:
        config += " -c textord_old_baselines=0"
    if vertical:
        config += " -c textord_old_baselines=0"
    text = image_to_string(image, config=config)
    return text.strip()


def ocr_by_vision(image, media_language):
    languages = [vision_language_options[media_language]]  # 设置需要识别的语言

    height, width, _ = image.shape
    image_data = image.tobytes()
    provider = CGDataProviderCreateWithData(None, image_data, len(image_data), None)
    cg_image = CGImageCreate(width, height, 8, 8 * 3, width * 3, CGColorSpaceCreateDeviceRGB(), 0, provider, None,
                             False, kCGRenderingIntentDefault)

    handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

    request = VNRecognizeTextRequest.new()
    request.setRecognitionLevel_(0)  # 使用快速识别（0-快速，1-精确）
    request.setUsesLanguageCorrection_(True)  # 使用语言纠正
    request.setRecognitionLanguages_(languages)  # 设置识别的语言

    error = None
    success = handler.performRequests_error_([request], error)

    if success:
        results = request.results()
        text_results = []
        for result in results:
            text = result.text()
            confidence = result.confidence()
            bounding_box = result.boundingBox()
            x, y, w, h = bounding_box.origin.x, bounding_box.origin.y, bounding_box.size.width, bounding_box.size.height
            text_results.append((str(text), (x, y, w, h), confidence))
        return text_results
    else:
        print("Error: ", error)
        return []


def ocr_by_easyocr(image, media_language, vertical=False):
    reader = Reader([media_language], gpu=False)
    result = reader.readtext(image, detail=0, paragraph=True, width_ths=0.8, text_threshold=0.5, slope_ths=0.5,
                             ycenter_ths=0.5)
    recognized_text = ' '.join(result)
    return recognized_text.strip()


def ocr_by_baidu(image, media_language):
    # 初始化百度OCR客户端
    client = AipOcr(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

    # 将图像转换为base64编码
    _, encoded_image = imencode('.png', image)
    image_data = encoded_image.tobytes()
    base64_image = b64encode(image_data)

    # 设置百度OCR识别选项
    options = {'language_type': baidu_language_options[media_language]}

    # 调用百度OCR API进行文字识别
    result = client.basicGeneral(base64_image, options)

    # 提取识别结果
    recognized_text = ''
    if 'words_result' in result:
        words = [word['words'] for word in result['words_result']]
        recognized_text = ' '.join(words)

    return recognized_text.strip()


def get_comictextdetector_mask(image):
    ih, iw = image.shape[0:2]

    # 缩放并填充图像，同时满足步长倍数约束
    raw_shape = image.shape[:2]  # 当前形状 [高度, 宽度]

    # 缩放比例 (新 / 旧)
    r = min(input_tuple[0] / ih, input_tuple[1] / iw)
    if not scaleup:  # 只缩小，不放大（以获得更好的验证mAP）
        r = min(r, 1.0)

    # 计算填充
    new_unpad = int(round(iw * r)), int(round(ih * r))
    dw, dh = input_tuple[1] - new_unpad[0], input_tuple[0] - new_unpad[1]  # wh填充
    if auto:  # 最小矩形
        dw, dh = mod(dw, stride), mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (input_tuple[1], input_tuple[0])
        ratio = input_tuple[1] / iw, input_tuple[0] / ih  # 宽度，高度比例

    dh, dw = int(dh), int(dw)

    if raw_shape[::-1] != new_unpad:  # 调整大小
        image = resize(image, new_unpad, interpolation=INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = copyMakeBorder(image, 0, dh, 0, dw, BORDER_CONSTANT, value=color_black)  # 添加边框

    if to_tensor:
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = array([ascontiguousarray(image)]).astype(float32) / 255
        if to_tensor:
            image = torch.from_numpy(image).to(device)
            if half:
                image = image.half()

    blob = dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(input_size, input_size))
    comictextdetector_model.setInput(blob)
    blks, mask, lines_map = comictextdetector_model.forward(uoln)
    if mask.shape[1] == 2:  # 一些OpenCV版本的输出结果是颠倒的
        tmp = mask
        mask = lines_map
        lines_map = tmp
    mask = mask.squeeze()
    mask = mask[..., :mask.shape[0] - dh, :mask.shape[1] - dw]
    lines_map = lines_map[..., :lines_map.shape[2] - dh, :lines_map.shape[3] - dw]

    # img = img.permute(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze_()
        if mask.device != 'cpu':
            mask = mask.detach().cpu()
        mask = mask.numpy()
    else:
        mask = mask.squeeze()
    if thresh is not None:
        mask = mask > thresh
    mask = mask * 255
    # if isinstance(img, torch.Tensor):
    mask = mask.astype(uint8)

    # map output to input img
    mask = resize(mask, (iw, ih), interpolation=INTER_LINEAR)
    return mask


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
    image_pil = Image.fromarray(cvtColor(image_raw, COLOR_BGR2RGB))
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
        text_bbox = draw.textbbox((cx, cy), text, font=font100, anchor="mm")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text((cx - text_width // 2, cy - text_height // 2), text, font=font100, fill=semi_transparent_purple)

    # 将带有半透明颜色轮廓的透明图像与原图混合
    image_pil.paste(overlay, mask=overlay)
    # 将 PIL 图像转换回 OpenCV 图像
    blended_image = cvtColor(array(image_pil), COLOR_RGB2BGR)
    return blended_image


@timer_decorator
@logger.catch
def get_raw_bubbles(bubble_mask, letter_mask, comictextdetector_mask):
    ih, iw = bubble_mask.shape[0:2]
    black_background = zeros((ih, iw), dtype=uint8)

    all_contours, hierarchy = findContours(bubble_mask, RETR_TREE, CHAIN_APPROX_SIMPLE)

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
            filled_contour = drawContours(black_background.copy(), [cnt.contour], 0, color=255, thickness=-1)

            # 使用位运算将 mask 和 filled_contour 相交，得到轮廓内的白色像素
            bubble_in_contour = bitwise_and(bubble_mask, filled_contour)
            letter_in_contour = bitwise_and(letter_mask, filled_contour)
            bubble_px = np.sum(bubble_in_contour == 255)
            letter_px = np.sum(letter_in_contour == 255)

            # ================对轮廓数值进行初筛================
            bubble_px_ratio = bubble_px / cnt.area  # 气泡像素占比
            letter_px_ratio = letter_px / cnt.area  # 文字像素占比
            bubble_and_letter_px_ratio = bubble_px_ratio + letter_px_ratio

            # ================气泡最外圈区域文字像素数量================
            # 步骤1：使用erode函数缩小filled_contour，得到一个缩小的轮廓
            eroded_contour = erode(filled_contour, kernel5, iterations=1)
            # 步骤2：从原始filled_contour中减去eroded_contour，得到轮廓的边缘
            contour_edges = subtract(filled_contour, eroded_contour)
            # 步骤3：在letter_mask上统计边缘上的白色像素数量
            edge_pixels = where(contour_edges == 255)
            white_edge_pixel_count = 0
            for y, x in zip(*edge_pixels):
                if letter_mask[y, x] == 255:
                    white_edge_pixel_count += 1

            # ================对轮廓数值进行初筛================
            if comictextdetector_mask is not None:
                mask_in_contour = bitwise_and(comictextdetector_mask, filled_contour)
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
    black_background = zeros((ih, iw), dtype=uint8)
    zero_image = zeros_like(bubble_mask, dtype=int32)
    single_cnts = []
    for f in range(len(filtered_cnts)):
        filtered_cnt = filtered_cnts[f]
        filled_contour = drawContours(black_background.copy(), [filtered_cnt.contour], 0, color=255, thickness=-1)
        bubble_in_contour = bitwise_and(bubble_mask, filled_contour)
        letter_in_contour = bitwise_and(letter_mask, filled_contour)

        # 膨胀文字部分来估计实际气泡数
        dilate_contour = dilate(letter_in_contour, kernel15, iterations=1)

        bulk_contours, hierarchy = findContours(dilate_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
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
                color_labels = zeros((labels.shape[0], labels.shape[1], 3), dtype=uint8)
                # 为每个标签分配一种颜色
                unique_labels = unique(labels)
                colors = (colormap_tab20(arange(len(unique_labels)))[:, :3] * 255).astype(uint8)

                for label, color in zip(unique_labels, colors):
                    if label == 0:
                        continue
                    color_labels[labels == label] = color

                scnts = []
                for u in range(len(unique_labels)):
                    target_label = unique_labels[u]
                    binary_image = zeros_like(labels, dtype=uint8)
                    binary_image[labels == target_label] = 255

                    # 查找轮廓
                    sep_contours, hierarchy = findContours(binary_image, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.a0_para()
        self.a1_initialize()
        self.a2_status_bar()
        self.a3_docks()
        self.a4_menubar()
        self.a5_toolbar()
        self.a9_setting()

    def b1_window(self):
        return

    def a0_para(self):
        # ================初始化变量================
        self.screen_icon = icon('ei.screen')
        self.setWindowIcon(self.screen_icon)
        self.default_palette = QGuiApplication.palette()
        self.setAcceptDrops(True)

        self.setWindowTitle(window_title_prefix)
        self.resize(1200, 800)

        # 创建 QGraphicsScene 和 QGraphicsView 对象
        self.graphics_scene = QGraphicsScene(self)  # 图形场景对象
        self.graphics_view = QGraphicsView(self)  # 图形视图对象
        # 设置渲染、优化和视口更新模式
        # 设置渲染抗锯齿
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 设置优化标志以便不为抗锯齿调整
        self.graphics_view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # 设置优化标志以便不保存画家状态
        self.graphics_view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        # 设置视口更新模式为完整视口更新
        self.graphics_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # 设置渲染提示为平滑图像变换，以提高图像的显示质量
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def a1_initialize(self):
        # ================图片列表================
        self.image_folder = None
        self.image_list = []
        self.image_file = None
        self.image_item = None
        self.image = QImage()
        self.pixmap_item = QGraphicsPixmapItem()
        self.image_index = -1
        self.filtered_image_index = -1
        self.view_mode = 0
        self.screen_scaling_factor = self.get_screen_scaling_factor()
        self.screen_scaling_factor_reciprocal = 1 / self.screen_scaling_factor
        # ================最近文件夹================
        self.recent_folders = []
        # ================设置================
        # 根据需要修改组织和应用名
        self.program_settings = QSettings("MOMO", "MomoTranslator")
        self.media_type_index = self.program_settings.value('media_type_index', 0)
        self.media_type = media_type_dict[self.media_type_index]
        self.media_language_index = self.program_settings.value('media_language_index', 0)
        self.media_language = media_language_dict[self.media_language_index]

    def a2_status_bar(self):
        # ================状态栏================
        self.status_bar = QStatusBar()
        # 设置状态栏，类似布局设置
        self.setStatusBar(self.status_bar)

    def a3_docks(self):
        # ================缩略图列表================
        self.thumbnails_widget = QListWidget(self)
        self.thumbnails_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)  # 设置自定义右键菜单
        self.thumbnails_widget.customContextMenuRequested.connect(self.show_image_list_context_menu)  # 连接自定义右键菜单信号与函数
        self.thumbnails_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 设置单选模式
        self.thumbnails_widget.setViewMode(QListView.ViewMode.IconMode)  # 设置图标模式
        self.thumbnails_widget.setFlow(QListView.Flow.TopToBottom)  # 设置从上到下排列
        self.thumbnails_widget.setResizeMode(QListView.ResizeMode.Adjust)  # 设置自动调整大小
        self.thumbnails_widget.setWrapping(False)  # 关闭自动换行
        self.thumbnails_widget.setWordWrap(True)  # 开启单词换行
        self.thumbnails_widget.setIconSize(QSize(240, 240))  # 设置图标大小
        self.thumbnails_widget.setSpacing(5)  # 设置间距
        self.thumbnails_widget.itemSelectionChanged.connect(self.select_image_from_list)  # 连接图标选择信号与函数

        self.nav_tab = QTabWidget()  # 创建选项卡控件
        self.nav_tab.addTab(self.thumbnails_widget, self.tr('Thumbnails'))  # 添加缩略图列表到选项卡

        self.case_sensitive_button = QToolButton()  # 创建区分大小写按钮
        self.case_sensitive_button.setIcon(icon('msc.case-sensitive'))  # 设置图标
        self.case_sensitive_button.setCheckable(True)  # 设置可选中
        self.case_sensitive_button.setText(self.tr('Case Sensitive'))  # 设置文本
        self.case_sensitive_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)  # 设置仅显示图标

        self.whole_word_button = QToolButton()  # 创建全词匹配按钮
        self.whole_word_button.setIcon(icon('msc.whole-word'))  # 设置图标
        self.whole_word_button.setCheckable(True)  # 设置可选中
        self.whole_word_button.setText(self.tr('Whole Word'))  # 设置文本
        self.whole_word_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)  # 设置仅显示图标

        self.regex_button = QToolButton()  # 创建正则表达式按钮
        self.regex_button.setIcon(icon('mdi.regex'))  # 设置图标
        self.regex_button.setCheckable(True)  # 设置可选中
        self.regex_button.setText(self.tr('Use Regex'))  # 设置文本
        self.regex_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)  # 设置仅显示图标

        # 连接按钮的点击事件与搜索结果的刷新
        self.case_sensitive_button.clicked.connect(self.refresh_search_results)
        self.whole_word_button.clicked.connect(self.refresh_search_results)
        self.regex_button.clicked.connect(self.refresh_search_results)

        self.hb_search_bar = QHBoxLayout()  # 创建水平布局，用于将三个按钮放在一行
        self.hb_search_bar.setContentsMargins(0, 0, 0, 0)
        self.hb_search_bar.addStretch()
        self.hb_search_bar.addWidget(self.case_sensitive_button)
        self.hb_search_bar.addWidget(self.whole_word_button)
        self.hb_search_bar.addWidget(self.regex_button)

        self.search_bar = QLineEdit()  # 创建搜索栏
        self.search_bar.setPlaceholderText(self.tr('Search'))  # 设置占位符文本
        self.search_bar.textChanged.connect(self.filter_image_list)  # 连接文本改变信号与函数
        self.search_bar.setLayout(self.hb_search_bar)  # 将按钮添加到 QLineEdit 的右侧

        self.vb_search_nav = QVBoxLayout()  # 创建垂直布局，用于将搜索框和图片列表放在一列
        self.vb_search_nav.addWidget(self.search_bar)
        self.vb_search_nav.addWidget(self.nav_tab)

        self.pics_widget = QWidget()  # 创建 QWidget，用于容纳搜索框和图片列表
        self.pics_widget.setLayout(self.vb_search_nav)

        self.pics_dock = QDockWidget(self.tr('Image List'), self)  # 创建 QDockWidget，用于显示图片列表
        self.pics_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.pics_dock.setWidget(self.pics_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pics_dock)

        # ================创建设置工具================
        self.lb_setting = QLabel('翻译设置')
        self.lb_setting.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.hb_media_type = QHBoxLayout()
        self.lb_media_type = QLabel(self.tr('Layout'))
        self.lb_media_type.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hb_media_type.addWidget(self.lb_media_type)

        # 创建选项组
        self.media_type_names = [
            self.tr('Comic'),
            self.tr('Manga'),
        ]
        self.bg_media_type = QButtonGroup()
        self.bg_media_type.buttonClicked.connect(self.update_media_type)
        for index, option in enumerate(self.media_type_names):
            self.rb_media_type = QRadioButton(option)
            self.hb_media_type.addWidget(self.rb_media_type)
            self.bg_media_type.addButton(self.rb_media_type, index)
            if index == self.media_type_index:
                self.rb_media_type.setChecked(True)

        self.hb_media_type.addStretch(1)

        # 创建语言选择下拉框
        self.cb_media_language = QComboBox()
        self.cb_media_language.addItems([
            self.tr("English"),
            self.tr("Chinese Simplified"),
            self.tr("Chinese Traditional"),
            self.tr("Japanese"),
            self.tr("Korean"),
        ])
        self.cb_media_language.setCurrentIndex(self.media_language_index)
        self.cb_media_language.currentIndexChanged.connect(self.update_media_language)

        # 在布局中添加下拉框
        self.hb_media_language = QHBoxLayout()
        self.lb_media_language = QLabel(self.tr('Language'))
        self.lb_media_language.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hb_media_language.addWidget(self.lb_media_language)
        self.hb_media_language.addWidget(self.cb_media_language)
        self.hb_media_language.addStretch(1)

        self.vb_setting = QVBoxLayout()
        self.vb_setting.addWidget(self.lb_setting)
        self.vb_setting.addLayout(self.hb_media_type)
        self.vb_setting.addLayout(self.hb_media_language)
        self.vb_setting.addStretch(1)
        self.setting_tool = QWidget()
        self.setting_tool.setLayout(self.vb_setting)

        # ================创建文本工具================
        self.lb_text = QLabel('为LabelPlus提供兼容')
        self.lb_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.vb_text = QVBoxLayout()
        self.vb_text.addWidget(self.lb_text)
        self.vb_text.addStretch(1)
        self.text_tool = QWidget()
        self.text_tool.setLayout(self.vb_text)

        # ================创建图层工具================
        self.lb_layer = QLabel('为PS提供兼容')
        self.lb_layer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.vb_layer = QVBoxLayout()
        self.vb_layer.addWidget(self.lb_layer)
        self.vb_layer.addStretch(1)
        self.layer_tool = QWidget()
        self.layer_tool.setLayout(self.vb_layer)

        # ================创建步骤工具================
        self.hb_step = QHBoxLayout()
        self.pb_step = QPushButton(self.tr('Start'))
        self.pb_step.clicked.connect(self.start_task)
        self.hb_step.addWidget(self.pb_step)

        # 创建选项组
        self.task_names = [
            self.tr('Analyze Frames'),
            self.tr('Analyze Bubbles'),
            self.tr('OCR'),
            self.tr('Translate'),
            self.tr('Lettering'),
        ]
        self.bg_step = QButtonGroup()
        for index, option in enumerate(self.task_names):
            self.rb_task = QRadioButton(option)
            self.hb_step.addWidget(self.rb_task)
            self.bg_step.addButton(self.rb_task, index)
            # 默认选中第一个 QRadioButton
            if index == 0:
                self.rb_task.setChecked(True)

        # 创建进度条
        self.pb_task = QProgressBar()
        self.hb_step.addWidget(self.pb_task)
        self.step_tool = QWidget()
        self.step_tool.setLayout(self.hb_step)

        self.setting_dock = QDockWidget(self.tr('Setting'), self)
        self.setting_dock.setObjectName("SettingDock")
        self.setting_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setting_dock.setWidget(self.setting_tool)
        self.setting_dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.setting_dock)

        self.text_dock = QDockWidget(self.tr('Text'), self)
        self.text_dock.setObjectName("TextDock")
        self.text_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.text_dock.setWidget(self.text_tool)
        self.text_dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.text_dock)

        self.layer_dock = QDockWidget(self.tr('Layer'), self)
        self.layer_dock.setObjectName("LayerDock")
        self.layer_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.layer_dock.setWidget(self.layer_tool)
        self.layer_dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_dock)

        self.step_dock = QDockWidget(self.tr('Step'), self)
        self.step_dock.setObjectName("StepDock")
        self.step_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.step_dock.setWidget(self.step_tool)
        self.step_dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.step_dock)

    def a4_menubar(self):
        # 文件菜单
        self.file_menu = self.menuBar().addMenu(self.tr('File'))

        # 添加打开文件夹操作
        self.open_folder_action = QAction(self.tr('Open Folder'), self)
        self.open_folder_action.setIcon(icon('ei.folder'))
        self.open_folder_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_folder_action.triggered.connect(self.open_folder_by_dialog)
        self.file_menu.addAction(self.open_folder_action)

        # 添加保存图片操作
        self.save_image_action = QAction(self.tr('Save Image'), self)
        self.save_image_action.setIcon(icon('ri.image-edit-fill'))
        self.save_image_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_image_action.triggered.connect(self.save_image)
        self.file_menu.addAction(self.save_image_action)

        self.file_menu.addSeparator()

        # 添加最近打开文件夹菜单项
        self.recent_folders_menu = self.file_menu.addMenu(self.tr('Recent Folders'))
        self.update_recent_folders_menu()

        # 显示菜单
        self.view_menu = self.menuBar().addMenu(self.tr('View'))

        # 添加视图菜单选项
        self.view_menu.addAction(self.text_dock.toggleViewAction())
        self.view_menu.addAction(self.layer_dock.toggleViewAction())
        self.view_menu.addAction(self.pics_dock.toggleViewAction())
        self.view_menu.addSeparator()

        # 添加缩放选项
        self.zoom_in_action = QAction(self.tr('Zoom In'), self)
        self.zoom_in_action.setIcon(icon('ei.zoom-in'))
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.view_menu.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(self.tr('Zoom Out'), self)
        self.zoom_out_action.setIcon(icon('ei.zoom-out'))
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.view_menu.addAction(self.zoom_out_action)

        self.view_menu.addSeparator()

        # 添加自适应屏幕选项
        self.fit_to_screen_action = QAction(self.tr('Fit to Screen'), self)
        self.fit_to_screen_action.setIcon(icon('mdi6.fit-to-screen-outline'))
        self.fit_to_screen_action.setShortcut(QKeySequence("Ctrl+F"))
        self.fit_to_screen_action.setCheckable(True)  # 将选项设置为可选
        self.fit_to_screen_action.toggled.connect(self.fit_to_view_toggled)
        self.view_menu.addAction(self.fit_to_screen_action)

        # 添加自适应宽度选项
        self.fit_to_width_action = QAction(self.tr('Fit to Width'), self)
        self.fit_to_width_action.setIcon(icon('ei.resize-horizontal'))
        self.fit_to_width_action.setShortcut(QKeySequence("Ctrl+W"))
        self.fit_to_width_action.setCheckable(True)  # 将选项设置为可选
        self.fit_to_width_action.toggled.connect(self.fit_to_view_toggled)
        self.view_menu.addAction(self.fit_to_width_action)

        # 添加自适应高度选项
        self.fit_to_height_action = QAction(self.tr('Fit to Height'), self)
        self.fit_to_height_action.setIcon(icon('ei.resize-vertical'))
        self.fit_to_height_action.setShortcut(QKeySequence("Ctrl+H"))
        self.fit_to_height_action.setCheckable(True)  # 设置选项为可选
        self.fit_to_height_action.toggled.connect(self.fit_to_view_toggled)
        self.view_menu.addAction(self.fit_to_height_action)

        # 添加重置缩放选项
        self.reset_zoom_action = QAction(self.tr('Reset Zoom'), self)
        self.reset_zoom_action.setIcon(icon('mdi6.backup-restore'))
        self.reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        self.reset_zoom_action.triggered.connect(self.reset_zoom)
        self.view_menu.addAction(self.reset_zoom_action)

        self.view_menu.addSeparator()

        # 添加显示选项
        self.display_modes = [(self.tr('Show Thumbnails'), 0),
                              (self.tr('Show Filenames'), 1),
                              (self.tr('Show Both'), 2)]
        self.display_mode_group = QActionGroup(self)
        for display_mode in self.display_modes:
            action = QAction(display_mode[0], self, checkable=True)
            action.triggered.connect(lambda _, mode=display_mode[1]: self.set_display_mode(mode))
            self.view_menu.addAction(action)
            self.display_mode_group.addAction(action)

        self.display_mode_group.actions()[2].setChecked(True)  # 默认选中 Show Both 选项

        # 编辑菜单
        self.edit_menu = self.menuBar().addMenu(self.tr('Edit'))

        # 导航菜单
        self.nav_menu = self.menuBar().addMenu(self.tr('Navigate'))

        # 添加上一张图片操作
        self.prev_image_action = QAction(self.tr('Previous Image'), self)
        self.prev_image_action.setIcon(icon('ei.arrow-left'))
        self.prev_image_action.setShortcut(QKeySequence("Ctrl+Left"))
        self.prev_image_action.triggered.connect(lambda: self.change_image(-1))
        self.nav_menu.addAction(self.prev_image_action)

        # 添加下一张图片操作
        self.next_image_action = QAction(self.tr('Next Image'), self)
        self.next_image_action.setIcon(icon('ei.arrow-right'))
        self.next_image_action.setShortcut(QKeySequence("Ctrl+Right"))
        self.next_image_action.triggered.connect(lambda: self.change_image(1))
        self.nav_menu.addAction(self.next_image_action)

        # 添加第一张图片操作
        self.first_image_action = QAction(self.tr('First Image'), self)
        self.first_image_action.setIcon(icon('ei.step-backward'))
        self.first_image_action.setShortcut(QKeySequence("Ctrl+Home"))
        self.first_image_action.triggered.connect(lambda: self.change_image("first"))
        self.nav_menu.addAction(self.first_image_action)

        # 添加最后一张图片操作
        self.last_image_action = QAction(self.tr('Last Image'), self)
        self.last_image_action.setIcon(icon('ei.step-forward'))
        self.last_image_action.setShortcut(QKeySequence("Ctrl+End"))
        self.last_image_action.triggered.connect(lambda: self.change_image("last"))
        self.nav_menu.addAction(self.last_image_action)

        # 帮助菜单
        self.help_menu = self.menuBar().addMenu(self.tr('Help'))

        self.about_action = QAction(f"{self.tr('About')} {APP_NAME}", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.about_action)

        self.about_qt_action = QAction(f"{self.tr('About')} Qt", self)
        self.about_qt_action.triggered.connect(QApplication.instance().aboutQt)
        self.help_menu.addAction(self.about_qt_action)

        self.help_document_action = QAction(f"{APP_NAME} {self.tr('Help')}", self)
        self.help_document_action.triggered.connect(self.show_help_document)
        self.help_menu.addAction(self.help_document_action)

        self.feedback_action = QAction(self.tr('Bug Report'), self)
        self.feedback_action.triggered.connect(self.show_feedback_dialog)
        self.help_menu.addAction(self.feedback_action)

        self.update_action = QAction(self.tr('Update Online'), self)
        self.update_action.triggered.connect(self.check_for_updates)
        self.help_menu.addAction(self.update_action)

    def a5_toolbar(self):
        # 添加顶部工具栏
        self.tool_bar = QToolBar(self.tr('Toolbar'), self)
        self.tool_bar.setObjectName("Toolbar")
        self.tool_bar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)
        self.tool_bar.setMovable(False)
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.tool_bar.addAction(self.open_folder_action)
        self.tool_bar.addAction(self.save_image_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.zoom_in_action)
        self.tool_bar.addAction(self.zoom_out_action)
        self.tool_bar.addAction(self.fit_to_screen_action)
        self.tool_bar.addAction(self.fit_to_width_action)
        self.tool_bar.addAction(self.fit_to_height_action)
        self.tool_bar.addAction(self.reset_zoom_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.first_image_action)
        self.tool_bar.addAction(self.prev_image_action)
        self.tool_bar.addAction(self.next_image_action)
        self.tool_bar.addAction(self.last_image_action)

        # 添加缩放百分比输入框
        self.scale_percentage_edit = QLineEdit(self)
        self.scale_percentage_edit.setFixedWidth(60)
        self.scale_percentage_edit.setValidator(QDoubleValidator(1, 1000, 2))
        self.scale_percentage_edit.setText('100.00')
        self.scale_percentage_edit.editingFinished.connect(self.scale_by_percentage)
        self.tool_bar.addWidget(self.scale_percentage_edit)
        self.tool_bar.addWidget(QLabel('%'))

    def a9_setting(self):
        # 读取上一次打开的文件夹、窗口位置和状态
        last_opened_folder = self.program_settings.value('last_opened_folder', '')
        geometry = self.program_settings.value('window_geometry')
        state = self.program_settings.value('window_state')
        # 如果上一次打开的文件夹存在，则打开它
        if last_opened_folder and os.path.exists(last_opened_folder) and os.path.isdir(last_opened_folder):
            self.open_folder_by_path(last_opened_folder)
        # 如果上一次有记录窗口位置，则恢复窗口位置
        if geometry is not None:
            self.restoreGeometry(geometry)
        # 如果上一次有记录窗口状态，则恢复窗口状态
        if state is not None:
            self.restoreState(state)

        # 将 QGraphicsView 设置为中心窗口部件
        self.setCentralWidget(self.graphics_view)

        # 显示窗口
        self.show()

    def _display_image(self, pixmap):
        self.image_size = pixmap.size()

        # ================清除之前的图像================
        if self.image_item:
            self.graphics_scene.removeItem(self.image_item)  # 移除之前的图片项

        # ================显示新图片================
        self.image_item = self.graphics_scene.addPixmap(pixmap)
        # 将视图大小设置为 pixmap 的大小，并将图像放入视图中
        self.graphics_view.setSceneRect(pixmap.rect().toRectF())
        self.graphics_view.setScene(self.graphics_scene)

        # 设置视图渲染选项
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.graphics_view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 设置视图转换和缩放选项
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setBackgroundBrush(QBrush(Qt.GlobalColor.lightGray))
        self.graphics_view.setTransform(
            QTransform().scale(self.screen_scaling_factor_reciprocal, self.screen_scaling_factor_reciprocal))

        # 更新缩放比例
        self.update_scale_percentage()

        # 如果自适应屏幕选项已经选中，则按照自适应屏幕缩放图片
        if self.fit_to_screen_action.isChecked():
            self.fit_to_view("screen")
        # 如果自适应宽度选项已经选中，则按照自适应宽度缩放图片
        elif self.fit_to_width_action.isChecked():
            self.fit_to_view("width")
        # 如果自适应高度选项已经选中，则按照自适应宽度缩放图片
        elif self.fit_to_height_action.isChecked():
            self.fit_to_view("height")

    def open_image_from_data(self, image_data, bgr_order=False):
        # 如果输入是Pillow图像，将其转换为NumPy数组
        if isinstance(image_data, Image.Image):
            image_data = array(image_data)

        # 确保输入数据是NumPy数组
        if isinstance(image_data, ndarray):
            height, width, channel = image_data.shape
            bytes_per_line = channel * width

            # 如果输入图像使用BGR顺序，交换颜色通道以获得正确的RGB顺序
            if bgr_order:
                image_data = cvtColor(image_data, COLOR_BGR2RGB)

            # 如果输入图像有4个通道（带有Alpha通道），则使用QImage.Format_ARGB32
            if channel == 4:
                qimage_format = QImage.Format_ARGB32
            # 如果输入图像有3个通道，使用QImage.Format_RGB888
            elif channel == 3:
                qimage_format = QImage.Format_RGB888
            # 其他情况暂不支持
            else:
                raise ValueError("Unsupported number of channels in the input image.")

            # 将NumPy数组转换为QImage
            qimage = QImage(image_data.data, width, height, bytes_per_line, qimage_format)

            # 将QImage转换为QPixmap
            pixmap = QPixmap.fromImage(qimage)

            # 显示图像
            self._display_image(pixmap)
        else:
            raise ValueError("Input data must be a NumPy array or a Pillow Image.")
        QApplication.processEvents()

    def open_image_by_path(self, image_file):
        # 需要路径存在
        image_file = Path(image_file)
        if image_file.exists():
            self.image_file = image_file
            self.image_file_size = os.path.getsize(self.image_file)
            self.image_index = self.image_list.index(self.image_file)

            # ================显示新图片================
            pixmap = QPixmap(self.image_file.as_posix())
            self._display_image(pixmap)

            # 将当前图片项设为选中状态，更新状态栏信息
            self.thumbnails_widget.setCurrentRow(self.image_index)
            index_str = f'{self.image_index + 1}/{len(self.image_list)}'
            meta_str = f'{self.tr("Width")}: {self.image_size.width()} {self.tr("Height")}: {self.image_size.height()} | {self.tr("File Size")}: {self.image_file_size} bytes'
            status_text = f'{index_str} | {self.tr("Filnename")}: {self.image_file.name} | {meta_str}'
            self.status_bar.showMessage(status_text)

    def open_folder_by_path(self, folder_path):
        # 打开上次关闭程序时用到的文件夹
        # 打开最近的文件夹
        # 打开文件夹

        # 判断文件夹路径是否存在
        if os.path.exists(folder_path):
            # 获取所有图片文件的路径
            image_list = get_valid_image_list(folder_path)
            if image_list:
                self.image_folder = folder_path
                self.image_list = image_list
                self.filtered_image_list = self.image_list
                self.image_index = 0
                self.filtered_image_index = 0
                self.image_file = self.image_list[self.image_index]

                # ================更新导航栏中的图片列表================
                self.thumbnails_widget.clear()
                for image_file in self.image_list:
                    # 将image的basename作为文本添加到thumbnail_item中
                    thumbnail_item = QListWidgetItem(image_file.name)
                    # 将image的图标设置为缩略图
                    thumbnail_item.setIcon(QIcon(image_file.as_posix()))
                    # 文本居中显示
                    thumbnail_item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
                    thumbnail_item.setData(Qt.ItemDataRole.UserRole, image_file)
                    # 如果image文件存在，则将thumbnail_item添加到thumbnails_widget中
                    if image_file.exists():
                        self.thumbnails_widget.addItem(thumbnail_item)

                self.open_image_by_path(self.image_file)

                self.setWindowTitle(f'{window_title_prefix} - {self.image_folder}')
                # 将该文件夹添加到最近使用文件夹列表
                self.add_recent_folder(self.image_folder)

    def open_folder_by_dialog(self):
        # 如果self.image_folder已经设置，使用其上一级目录作为起始目录，否则使用当前目录
        self.image_folder = Path(self.image_folder) if self.image_folder else None
        start_directory = self.image_folder.parent.as_posix() if self.image_folder else "."
        image_folder = QFileDialog.getExistingDirectory(self, self.tr('Open Folder'), start_directory)
        if image_folder:
            self.image_folder = image_folder
            self.open_folder_by_path(self.image_folder)

    def update_recent_folders_menu(self):
        self.recent_folders_menu.clear()
        recent_folders = self.program_settings.value("recent_folders", [])
        for folder in recent_folders:
            action = QAction(str(folder), self)
            action.triggered.connect(lambda checked, p=folder: self.open_folder_by_path(p))
            self.recent_folders_menu.addAction(action)

    def add_recent_folder(self, folder_path):
        recent_folders = self.program_settings.value("recent_folders", [])
        if folder_path in recent_folders:
            recent_folders.remove(folder_path)
        recent_folders.insert(0, folder_path)
        recent_folders = recent_folders[:10]  # 保留最多10个最近文件夹

        self.program_settings.setValue("recent_folders", recent_folders)
        self.update_recent_folders_menu()

    def set_display_mode(self, mode):
        self.view_mode = mode
        self.thumbnails_widget.setUpdatesEnabled(False)
        for index in range(self.thumbnails_widget.count()):
            item = self.thumbnails_widget.item(index)
            data = item.data(Qt.ItemDataRole.UserRole)
            if self.view_mode == 0:  # Show thumbnails only
                item.setIcon(QIcon(data.as_posix()))
                item.setText('')
                self.thumbnails_widget.setWordWrap(False)
            elif self.view_mode == 1:  # Show filenames only
                item.setIcon(QIcon())  # Clear the icon
                item.setText(data.name)
                self.thumbnails_widget.setWordWrap(False)  # Make sure filenames don't wrap
            elif self.view_mode == 2:  # Show both thumbnails and filenames
                item.setIcon(QIcon(data.as_posix()))
                item.setText(data.name)
                self.thumbnails_widget.setWordWrap(True)

        thumbnail_size = QSize(240, 240)  # Set the thumbnail size to 240x240
        self.thumbnails_widget.setIconSize(thumbnail_size)  # Set the icon size

        if self.view_mode == 1:  # Show filenames only
            self.thumbnails_widget.setGridSize(QSize(-1, -1))  # Use default size for grid
        else:
            # Set grid size for thumbnails and both modes
            self.thumbnails_widget.setGridSize(
                QSize(thumbnail_size.width() + 30, -1))  # Add extra width for file names and spacing

        self.thumbnails_widget.setUpdatesEnabled(True)

    def select_image_from_list(self):
        # 通过键鼠点击获取的数据是真实序数image_index
        active_list_widget = self.nav_tab.currentWidget()
        selected_items = active_list_widget.selectedItems()
        if not selected_items:
            return
        current_item = selected_items[0]
        image_index = active_list_widget.row(current_item)
        if image_index != self.image_index:
            self.image_index = image_index
            image_file = current_item.data(Qt.ItemDataRole.UserRole)
            self.open_image_by_path(image_file)

    def open_image_in_viewer(self, file_path):
        if sys.platform == 'win32':
            os.startfile(os.path.normpath(file_path))
        elif sys.platform == 'darwin':
            Popen(['open', file_path])
        else:
            Popen(['xdg-open', file_path])

    def open_file_in_explorer(self, file_path):
        folder_path = os.path.dirname(file_path)
        if sys.platform == 'win32':
            Popen(f'explorer /select,"{os.path.normpath(file_path)}"')
        elif sys.platform == 'darwin':
            Popen(['open', '-R', file_path])
        else:
            Popen(['xdg-open', folder_path])

    def open_image_in_ps(self, file_path):
        if sys.platform == 'win32':
            photoshop_executable_path = "C:/Program Files/Adobe/Adobe Photoshop CC 2019/Photoshop.exe"  # 请根据您的Photoshop安装路径进行修改
            Popen([photoshop_executable_path, file_path])
        elif sys.platform == 'darwin':
            photoshop_executable_path = "/Applications/Adobe Photoshop 2021/Adobe Photoshop 2021.app"  # 修改此行
            Popen(['open', '-a', photoshop_executable_path, file_path])
        else:
            QMessageBox.warning(self, "Warning", "This feature is not supported on this platform.")

    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def show_image_list_context_menu(self, point):
        item = self.thumbnails_widget.itemAt(point)
        if item:
            context_menu = QMenu(self)
            open_file_action = QAction(self.tr('Open in Explorer'), self)
            open_file_action.triggered.connect(
                lambda: self.open_file_in_explorer(item.data(Qt.ItemDataRole.UserRole)))
            open_image_action = QAction(self.tr('Open in Preview'), self)
            open_image_action.triggered.connect(
                lambda: self.open_image_in_viewer(item.data(Qt.ItemDataRole.UserRole)))
            context_menu.addAction(open_file_action)
            context_menu.addAction(open_image_action)

            # 添加一个打开方式子菜单
            open_with_menu = context_menu.addMenu(self.tr('Open with'))
            open_with_ps = QAction(self.tr('Photoshop'), self)
            open_with_menu.addAction(open_with_ps)

            # 添加拷贝图片路径、拷贝图片名的选项
            copy_image_path = QAction(self.tr('Copy Image Path'), self)
            copy_image_name = QAction(self.tr('Copy Image Name'), self)

            context_menu.addAction(copy_image_path)
            context_menu.addAction(copy_image_name)

            # 连接触发器
            open_with_ps.triggered.connect(lambda: self.open_image_in_ps(item.data(Qt.ItemDataRole.UserRole)))
            copy_image_path.triggered.connect(
                lambda: self.copy_to_clipboard(item.data(Qt.ItemDataRole.UserRole).as_posix()))
            copy_image_name.triggered.connect(lambda: self.copy_to_clipboard(item.text()))
            context_menu.exec(self.thumbnails_widget.mapToGlobal(point))

    def refresh_search_results(self):
        # 用于三个筛选按钮
        search_text = self.search_bar.text()
        self.filter_image_list(search_text)

    def filter_image_list(self, search_text):
        # 用于搜索框内容更新
        case_sensitive = self.case_sensitive_button.isChecked()
        whole_word = self.whole_word_button.isChecked()
        use_regex = self.regex_button.isChecked()

        flags = re.IGNORECASE if not case_sensitive else 0

        if use_regex:
            if whole_word:
                search_text = fr"\b{search_text}\b"
            try:
                regex = re.compile(search_text, flags)
            except re.error:
                return
        else:
            if whole_word:
                search_text = fr"\b{re.escape(search_text)}\b"
            else:
                search_text = re.escape(search_text)
            regex = re.compile(search_text, flags)

        # ================筛选列表更新================
        self.filtered_image_list = []
        for image_file in self.image_list:
            match = regex.search(image_file.name)
            if match:
                self.filtered_image_list.append(image_file)

        # ================缩略图列表更新================
        for index in range(self.thumbnails_widget.count()):
            item = self.thumbnails_widget.item(index)
            item_text = item.text()
            match = regex.search(item_text)
            if match:
                item.setHidden(False)
            else:
                item.setHidden(True)

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(self, self.tr('Save Image'), "", "Images (*.png *.xpm *.jpg)")

        if file_name:
            image = QImage(self.graphics_scene.sceneRect().size().toSize(), QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)

            painter = QPainter(image)
            self.graphics_scene.render(painter)
            painter.end()

            image.save(file_name)

    def change_image(self, step):
        current_image_path = self.image_list[self.image_index]
        # 检查当前图片路径是否在过滤后的图片列表中
        if current_image_path not in self.filtered_image_list:
            return
        current_filtered_index = self.filtered_image_list.index(current_image_path)

        if step == "first":
            new_filtered_index = 0
        elif step == "last":
            new_filtered_index = len(self.filtered_image_list) - 1
        else:
            new_filtered_index = current_filtered_index + step

        if 0 <= new_filtered_index < len(self.filtered_image_list):
            new_image_path = self.filtered_image_list[new_filtered_index]
            self.open_image_by_path(new_image_path)

    def get_screen_scaling_factor(self):
        screen = QApplication.primaryScreen()
        if sys.platform == 'darwin':  # 如果是 MacOS 系统
            return screen.devicePixelRatio()
        else:
            return 1

    def zoom(self, scale_factor):
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        current_transform = self.graphics_view.transform()
        current_scale = current_transform.m11()
        new_scale = current_scale * scale_factor

        if 0.01 <= new_scale <= 10:
            current_transform.scale(scale_factor, scale_factor)
            self.graphics_view.setTransform(current_transform)
            self.update_scale_percentage()

    def zoom_in(self):
        self.zoom(1.2)

    def zoom_out(self):
        self.zoom(1 / 1.2)

    def fit_to_view(self, mode):
        if self.image_item:
            if mode == "screen":
                self.graphics_view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            elif mode == "width":
                view_width = self.graphics_view.viewport().width()
                pixmap_width = self.image_item.pixmap().width()
                scale_factor = view_width / pixmap_width
                self.graphics_view.setTransform(QTransform().scale(scale_factor, scale_factor))
            elif mode == "height":
                view_height = self.graphics_view.viewport().height()
                pixmap_height = self.image_item.pixmap().height()
                scale_factor = view_height / pixmap_height
                self.graphics_view.setTransform(QTransform().scale(scale_factor, scale_factor))
            self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.update_scale_percentage()

    def fit_to_view_toggled(self, checked):
        sender = self.sender()
        if sender == self.fit_to_screen_action:
            mode = 'screen'
            if checked:
                self.fit_to_width_action.setChecked(False)
                self.fit_to_height_action.setChecked(False)
        elif sender == self.fit_to_width_action:
            mode = 'width'
            if checked:
                self.fit_to_screen_action.setChecked(False)
                self.fit_to_height_action.setChecked(False)
        else:  # if sender == self.fit_to_height_action
            mode = 'height'
            if checked:
                self.fit_to_screen_action.setChecked(False)
                self.fit_to_width_action.setChecked(False)
        self.fit_to_view(mode)

    def reset_zoom(self):
        self.fit_to_screen_action.setChecked(False)
        self.fit_to_width_action.setChecked(False)
        self.fit_to_height_action.setChecked(False)
        # 使用 setTransform 代替 resetMatrix 并应用缩放因子
        self.graphics_view.setTransform(
            QTransform().scale(self.screen_scaling_factor_reciprocal, self.screen_scaling_factor_reciprocal))
        self.update_scale_percentage()

    def update_scale_percentage(self):
        current_scale = round(self.graphics_view.transform().m11() * 100, 2)
        self.scale_percentage_edit.setText(str(current_scale))

    def scale_by_percentage(self):
        scale_percentage = float(self.scale_percentage_edit.text())
        target_scale = scale_percentage / 100
        current_scale = self.graphics_view.transform().m11()

        scale_factor = target_scale / current_scale
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.scale(scale_factor, scale_factor)

    def show_about_dialog(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle(f"{self.tr('About')} {APP_NAME}")

        app_name_label = QLabel(APP_NAME)
        app_name_label.setFont(QFont("Arial", 20, QFont.Bold))

        version_label = QLabel(f"版本: {APP_VERSION}")
        libraries_label = QLabel("使用的库：PyQt6")
        license_label = QLabel("许可证: MIT License")

        main_layout = QVBoxLayout()
        main_layout.addWidget(app_name_label)
        main_layout.addWidget(version_label)
        main_layout.addWidget(libraries_label)
        main_layout.addWidget(license_label)
        about_dialog.setLayout(main_layout)
        about_dialog.exec_()

    def show_help_document(self):
        # 在此处显示帮助文档
        pass

    def show_feedback_dialog(self):
        recipient_email = "annebing@qq.com"  # 替换为您的电子邮件地址
        subject = "Bug Report - 漫画翻译工具"
        body = "请在此处详细描述您遇到的问题：\n\n\n\n软件版本：1.0.0"
        mailto_url = f"mailto:{recipient_email}?subject={subject}&body={body}"

        try:
            webbrowser.open(mailto_url)
        except webbrowser.Error as e:
            QMessageBox.warning(self, "错误", f"无法打开邮件客户端：{e}")

    def check_for_updates(self):
        # 在此处实现在线更新软件的功能
        pass

    @timer_decorator
    @logger.catch
    def step0_analyze_frames(self):
        if self.frame_yml_path.exists():
            with open(self.frame_yml_path, 'r') as yml_file:
                image_data = yaml.safe_load(yml_file)
        else:
            image_data = {}
        # ================分析画格================
        total_images = len(self.image_list)
        processed_images = 0

        for p, image_file in enumerate(self.image_list):
            logger.warning(f'{image_file=}')
            image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
            if image_file.name in image_data:
                frame_grid_strs = image_data[image_file.name]
            else:
                # 获取RGBA格式的边框像素
                edge_pixels_rgba = get_edge_pixels_rgba(image_raw)
                # 寻找主要颜色
                color_ratio, dominant_color = find_dominant_color(edge_pixels_rgba)
                # 获取颜色名称
                color_name = get_color_name(dominant_color)
                logger.info(
                    f"边框颜色中出现次数最多的颜色：{dominant_color}, 颜色名称：{color_name}, {color_ratio=}")
                # 计算主要颜色遮罩
                frame_mask = compute_frame_mask(image_raw, dominant_color)
                ih, iw = frame_mask.shape[0:2]
                rec0 = (0, 0, iw, ih)
                pg = PicGrid(frame_mask, self.media_type)
                grids = []
                if pg.basic:
                    rec0 = Rect(rec0, self.media_type)
                    rec0.get_devide(frame_mask)
                    grids.append(rec0)
                else:
                    rects = pg.rects_dic[pg.judgement]
                    for rect in rects:
                        get_frame_grid_recursive(rect, grids, frame_mask, self.media_type)
                frame_grid_strs = [rect.frame_grid_str for rect in grids]
                image_data[image_file.name] = frame_grid_strs

            processed_images += 1
            self.pb_task.setValue(int(processed_images / total_images * 100))
            QApplication.processEvents()

            pprint(frame_grid_strs)
            if len(frame_grid_strs) >= 2:
                marked_frames = get_marked_frames(frame_grid_strs, image_raw)
                self.open_image_from_data(marked_frames, bgr_order=True)

        # Update progress bar to 100%
        self.pb_task.setValue(100)
        QApplication.processEvents()

        # Sort image_data dictionary by image file names
        image_data_sorted = {k: image_data[k] for k in natsorted(image_data)}

        # Save the image_data_sorted dictionary to the yml file
        with open(self.frame_yml_path, 'w') as yml_file:
            yaml.dump(image_data_sorted, yml_file)

    @timer_decorator
    @logger.catch
    def step1_analyze_bubbles(self):
        auto_subdir = Auto / self.image_folder.name
        make_dir(auto_subdir)

        if self.frame_yml_path.exists():
            with open(self.frame_yml_path, 'r') as yml_file:
                image_data = yaml.safe_load(yml_file)
        else:
            image_data = {}
        total_images = len(self.image_list)
        processed_images = 0
        all_masks_old = get_valid_image_list(self.image_folder, mode='mask')

        for p, image_file in enumerate(self.image_list):
            logger.warning(f'{image_file=}')
            image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
            ih, iw = image_raw.shape[0:2]
            # ================矩形画格信息================
            if image_file.name in image_data:
                frame_grid_strs = image_data[image_file.name]
            else:
                frame_grid_strs = [f'0,0,{iw},{ih}~0,0,{iw},{ih}']
            # ================模型检测文字，文字显示为白色================
            if use_torch and comictextdetector_model is not None:
                comictextdetector_mask = get_comictextdetector_mask(image_raw)
            else:
                comictextdetector_mask = None
            # ================针对每一张图================
            for c in range(len(color_patterns)):
                # ================遍历每种气泡文字颜色组合================
                color_pattern = color_patterns[c]
                cp_bubble, cp_letter = color_pattern

                if isinstance(cp_bubble, list):
                    # 气泡为渐变色
                    color_bubble = ColorGradient(cp_bubble)
                else:
                    # 气泡为单色
                    color_bubble = Color(cp_bubble)

                if isinstance(cp_letter, list):
                    # 文字为双色
                    color_letter = ColorDouble(cp_letter)
                    color_letter_big = color_letter
                else:
                    # 文字为单色
                    color_letter = Color(cp_letter)
                    color_letter_big = Color(f'{cp_letter[:6]}-{max(color_letter.padding + 120, 180)}')

                if color_bubble.type == 'Color':
                    bubble_mask = color_bubble.get_img(image_raw)
                else:  # ColorGradient
                    img_tuple = color_bubble.get_img(image_raw)
                    bubble_mask, left_sample, right_sample = img_tuple
                letter_mask = color_letter_big.get_img(image_raw)

                filtered_cnts = get_raw_bubbles(bubble_mask, letter_mask, comictextdetector_mask)
                colorful_raw_bubbles = get_colorful_bubbles(image_raw, filtered_cnts)

                # ================切割相连气泡================
                single_cnts = seg_bubbles(filtered_cnts, bubble_mask, letter_mask, self.media_type)
                # ================通过画格重新排序气泡框架================
                single_cnts_recs = []
                for f in range(len(frame_grid_strs)):
                    frame_grid_str = frame_grid_strs[f]
                    # 使用正则表达式提取字符串中的所有整数
                    int_values = list(map(int, findall(r'\d+', frame_grid_str)))
                    # 按顺序分配值
                    x, y, w, h, xx, yy, ww, hh = int_values
                    single_cnts_rec = []
                    for s in range(len(single_cnts)):
                        single_cnt = single_cnts[s]
                        if x <= single_cnt.cx <= x + w and y <= single_cnt.cy <= y + h:
                            single_cnts_rec.append(single_cnt)
                    if self.media_type == 'Manga':
                        ax = -1
                    else:
                        ax = 1
                    single_cnts_rec.sort(key=lambda x: ax * x.cx + x.cy)
                    single_cnts_recs.append(single_cnts_rec)
                single_cnts_ordered = list(chain(*single_cnts_recs))
                if len(single_cnts_ordered) >= 1:
                    colorful_single_bubbles = get_colorful_bubbles(image_raw, single_cnts_ordered)
                    self.open_image_from_data(colorful_single_bubbles, bgr_order=True)

                    alpha = 0.1
                    # 创建一个带有10%透明度原图背景的图像
                    transparent_image = zeros((image_raw.shape[0], image_raw.shape[1], 4), dtype=uint8)
                    transparent_image[..., :3] = image_raw
                    transparent_image[..., 3] = int(255 * alpha)
                    # 在透明图像上绘制contours，每个contour使用不同颜色

                    for s in range(len(single_cnts_ordered)):
                        bubble_cnt = single_cnts_ordered[s]
                        # 从 tab20 颜色映射中选择一个颜色
                        color = colormap_tab20(s % 20)[:3]
                        color_rgb = tuple(int(c * 255) for c in color)
                        color_bgra = color_rgb[::-1] + (255,)
                        drawContours(transparent_image, [bubble_cnt.contour], -1, color_bgra, -1)

                    self.open_image_from_data(transparent_image)
                    cp_preview_jpg = auto_subdir / f'{image_file.stem}-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.jpg'
                    cp_mask_cnt_pic = auto_subdir / f'{image_file.stem}-Mask-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.png'
                    write_pic(cp_preview_jpg, colorful_single_bubbles)
                    write_pic(cp_mask_cnt_pic, transparent_image)

            processed_images += 1
            self.pb_task.setValue(int(processed_images / total_images * 100))
            QApplication.processEvents()

        # ================搬运气泡蒙版================
        auto_all_masks = get_valid_image_list(auto_subdir, mode='mask')
        # 如果步骤开始前没有气泡蒙版
        if not all_masks_old:
            for mask_src in auto_all_masks:
                mask_dst = self.image_folder / mask_src.name
                copy2(mask_src, mask_dst)

        # Update progress bar to 100%
        self.pb_task.setValue(100)
        QApplication.processEvents()

    def step2_OCR(self):
        if self.frame_yml_path.exists():
            with open(self.frame_yml_path, 'r') as yml_file:
                image_data = yaml.safe_load(yml_file)
        else:
            image_data = {}
        # ================分析画格================
        total_images = len(self.image_list)
        processed_images = 0
        # ================气泡蒙版================
        all_masks = get_valid_image_list(self.image_folder, mode='mask')
        # ================DOCX文档================
        # 创建一个新的Document对象
        OCR_doc = Document()
        for i in range(len(self.image_list)):
            image_file = self.image_list[i]
            logger.warning(f'{image_file=}')

            OCR_doc.add_paragraph(image_file.stem)
            OCR_doc.add_paragraph('')

            image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
            ih, iw = image_raw.shape[0:2]

            # ================获取对应的文字图片================
            single_cnts = get_single_cnts(image_file, image_raw, all_masks)
            logger.debug(f'{len(single_cnts)=}')
            # ================矩形画格信息================
            if image_file.name in image_data:
                frame_grid_strs = image_data[image_file.name]
            else:
                frame_grid_strs = [f'0,0,{iw},{ih}~0,0,{iw},{ih}']
            # ================通过画格重新排序气泡框架================
            single_cnts_recs = []
            for f in range(len(frame_grid_strs)):
                frame_grid_str = frame_grid_strs[f]
                # 使用正则表达式提取字符串中的所有整数
                int_values = list(map(int, findall(r'\d+', frame_grid_str)))
                # 按顺序分配值
                x, y, w, h, xx, yy, ww, hh = int_values
                single_cnts_rec = []
                for s in range(len(single_cnts)):
                    single_cnt = single_cnts[s]
                    if x <= single_cnt.cx <= x + w and y <= single_cnt.cy <= y + h:
                        single_cnts_rec.append(single_cnt)
                if self.media_type == 'Manga':
                    ax = -1
                else:
                    ax = 1
                single_cnts_rec.sort(key=lambda x: ax * x.cx + x.cy)
                single_cnts_recs.append(single_cnts_rec)
            single_cnts_ordered = list(chain(*single_cnts_recs))

            if len(single_cnts_ordered) >= 1:
                colorful_single_bubbles = get_colorful_bubbles(image_raw, single_cnts_ordered)
                self.open_image_from_data(colorful_single_bubbles, bgr_order=True)

            for c in range(len(single_cnts_ordered)):
                single_cnt = single_cnts_ordered[c]
                # ================对裁剪后的图像进行文字识别================
                if SYSTEM in ['MAC', 'M1']:
                    # ================Vision================
                    recognized_text = ocr_by_vision(single_cnt.cropped_image, self.media_language)
                    lines = []
                    for text, (x, y, w, h), confidence in recognized_text:
                        # print(f"{text}[{confidence:.2f}] {x=:.2f}, {y=:.2f}, {w=:.2f}, {h=:.2f}")
                        lines.append(text)
                else:
                    # ================tesseract================
                    recognized_text = ocr_by_tesseract(single_cnt.cropped_image, self.media_language, vertical)
                    # 将recognized_text分成多行
                    lines = recognized_text.splitlines()
                # 去除空行
                non_empty_lines = [line for line in lines if line.strip()]
                # 将非空行合并为一个字符串
                cleaned_text = lf.join(non_empty_lines)
                print(cleaned_text)
                print('-' * 24)

                # ================将图片添加到docx文件中================
                cropped_image_pil = Image.fromarray(single_cnt.cropped_image)
                image_dpi = cropped_image_pil.info.get('dpi', (96, 96))[0]  # 获取图片的dpi，如果没有dpi信息，则使用默认值96
                with io.BytesIO() as temp_buffer:
                    cropped_image_pil.save(temp_buffer, format=image_format.upper())
                    temp_buffer.seek(0)
                    pic_width_inches = cropped_image_pil.width / image_dpi
                    OCR_doc.add_picture(temp_buffer, width=Inches(pic_width_inches))
                # ================将识别出的文字添加到docx文件中================
                OCR_doc.add_paragraph(cleaned_text)
                # 在图片和文字之间添加一个空行
                OCR_doc.add_paragraph('')

            # Update progress bar
            self.pb_task.setValue(int(processed_images / total_images * 100))
            # Process pending events to keep the UI responsive
            QApplication.processEvents()

            # Increment processed images counter
            processed_images += 1

        # ================保存为DOCX================
        OCR_doc.save(self.OCR_docx_path)
        # Update progress bar to 100%
        self.pb_task.setValue(100)

    @logger.catch
    @timer_decorator
    def step3_translate(self):
        target_language = 'zh-CN'

        # 打开docx文件
        OCR_doc = Document(self.OCR_docx_path)

        # 读取并连接文档中的所有段落文本
        full_text = []
        for para in OCR_doc.paragraphs:
            full_text.append(para.text)
            # logger.debug(f'{para.text=}')

        index_dict = {}
        last_ind = 0
        inds = []
        for i in range(len(self.image_list)):
            image_file = self.image_list[i]
            logger.warning(f'{image_file=}')
            if image_file.stem in full_text[last_ind:]:
                ind = full_text[last_ind:].index(image_file.stem) + last_ind
                index_dict[image_file.stem] = ind
                inds.append(ind)
                last_ind = ind
                logger.debug(f'{ind=}')
        inds.append(len(full_text))
        # pprint(index_dict)

        pin = 0
        para_dict = {}
        for i in range(len(self.image_list)):
            image_file = self.image_list[i]
            if image_file.stem in index_dict:
                start_i = inds[pin] + 1
                end_i = inds[pin + 1]
                pin += 1
                para_dict[image_file.stem] = full_text[start_i:end_i]
        # pprint(para_dict)

        all_valid_paras = []
        for key in para_dict:
            paras = para_dict[key]
            valid_paras = [x for x in paras if x.strip() != '']
            if valid_paras:
                all_valid_paras.extend(valid_paras)

        simple_lines = []
        for a in range(len(all_valid_paras)):
            para = all_valid_paras[a]
            # 将多行文本替换为单行文本
            single_line_text = para.replace('\n', ' ')
            # 将文本分割成句子
            sentences = re.split(r' *[\.\?!][\'"\)\]]* *', single_line_text)
            # 对每个句子进行首字母大写处理，同时考虑缩写
            capitalized_sentences = [capitalize_sentence(sentence) for sentence in sentences if sentence]
            # 将处理后的句子连接成一个字符串
            processed_text = '. '.join(capitalized_sentences) + '.'
            simple_lines.append(processed_text)

        simple_text = lf.join(simple_lines)
        print(simple_text)

        chunks = []
        current_chunk = ""
        for line in simple_lines:
            # 检查将当前行添加到当前块后的长度是否超过最大字符数
            if len(current_chunk) + len(line) + 1 > max_chars:  # 加1是为了考虑换行符
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += line + "\n"

        # 添加最后一个块（如果有内容的话）
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        for chunk in chunks:
            translated_text = GoogleTranslator(source='auto', target=target_language).translate(chunk)
            print(translated_text)

    def step4_lettering(self):
        pass

    def start_task(self, checked):
        # 当使用 QPushButton 的 clicked 信号连接到槽时，它会传递一个布尔值表示按钮是否被选中
        # 检查哪个 QRadioButton 被选中并执行相应的任务
        self.task_name = self.bg_step.checkedButton().text()
        self.task_ind = self.bg_step.checkedId()
        print(f'[{self.task_ind}]{self.task_name}')
        # 在这里执行您的任务逻辑
        if self.image_folder is not None:
            self.image_folder = Path(self.image_folder)
            if self.image_folder.exists():
                self.frame_yml_path = self.image_folder.parent / f'{self.image_folder.name}.yml'
                self.OCR_docx_path = self.image_folder.parent / f'{self.image_folder.name}-OCR.docx'
                if self.task_ind == 0:
                    self.step0_analyze_frames()
                elif self.task_ind == 1:
                    self.step1_analyze_bubbles()
                elif self.task_ind == 2:
                    self.step2_OCR()
                elif self.task_ind == 3:
                    self.step3_translate()
                elif self.task_ind == 4:
                    self.step4_lettering()

                # 自动选中下一个步骤，除非当前步骤是最后一个步骤
                if self.task_ind < self.bg_step.id(self.bg_step.buttons()[-1]):
                    next_task_ind = self.task_ind + 1
                    next_task_button = self.bg_step.button(next_task_ind)
                    if next_task_button:
                        next_task_button.setChecked(True)
                        self.task_ind = next_task_ind

        # Process pending events to keep the UI responsive
        QApplication.processEvents()

    def update_media_type(self, button):
        index = self.bg_media_type.id(button)
        self.media_type_index = index
        self.media_type = media_type_dict[index]

    def update_media_language(self, index):
        self.media_language_index = index
        self.media_language = media_language_dict[index]

    def closeEvent(self, event):
        # 如果有打开图片的文件夹，将其保存到程序设置中
        if self.image_folder:
            self.program_settings.setValue('last_opened_folder', self.image_folder)
        else:
            self.program_settings.setValue('last_opened_folder', '')
        # 保存窗口的几何和状态信息到程序设置中
        self.program_settings.setValue('window_geometry', self.saveGeometry())
        self.program_settings.setValue('window_state', self.saveState())
        self.program_settings.setValue('media_type_index', self.media_type_index)
        self.program_settings.setValue('media_language_index', self.media_language_index)
        event.accept()


@logger.catch
def main_qt():
    window = MainWindow()
    sys.exit(appgui.exec())


def self_translate_qt():
    src_head = ['Source', '目标语言']

    for i in range(len(language_tuples)):
        language_tuple = language_tuples[i]
        lang_code, en_name, cn_name, self_name = language_tuple

        ts_file = UserDataFolder / f'{APP_NAME}_{lang_code}.ts'
        csv_file = UserDataFolder / f'{APP_NAME}_{lang_code}.csv'

        # 提取可翻译的字符串生成ts文件
        cmd = f'{pylupdate} {py_path.as_posix()} -ts {ts_file.as_posix()}'
        logger.debug(f'{cmd}')
        call(cmd, shell=True)
        # 解析 ts 文件
        xml_text = read_txt(ts_file)
        doc_parse = xmltodict.parse(xml_text)

        # 使用预先提供的翻译更新 ts 文件
        messages = doc_parse['TS']['context']['message']

        en2dst, dst2en = {}, {}
        existing_sources = set()
        src = []
        if csv_file.exists():
            src = iread_csv(csv_file, True, False)
            for pind in range(len(src)):
                entry = src[pind]
                en_str = entry[0]
                dst_str = entry[-1]
                existing_sources.add(en_str)
                if en_str != '' and dst_str != '':
                    en2dst[en_str] = dst_str
                    dst2en[dst_str] = en_str

        updated_messages = deepcopy(messages)
        missing_sources = []
        for message in updated_messages:
            source = message['source']
            if source not in existing_sources:
                missing_sources.append(source)
            if source in en2dst:
                message['translation'] = en2dst[source]

        doc_parse['TS']['context']['message'] = updated_messages
        if missing_sources:
            new_src = deepcopy(src)
            for missing_source in missing_sources:
                new_src.append([missing_source, ''])
            write_csv(csv_file, new_src, src_head)

        # 保存更新后的 ts 文件
        with ts_file.open('w', encoding='utf-8') as f:
            f.write(xmltodict.unparse(doc_parse))

        # 生成 qm
        cmd = f'{lrelease} {ts_file.as_posix()}'
        call(cmd, shell=True)


@timer_decorator
def step0_analyze_frames():
    if frame_yml_path.exists():
        with open(frame_yml_path, 'r') as yml_file:
            image_data = yaml.safe_load(yml_file)
    else:
        image_data = {}
    # ================分析画格================
    for p, image_file in enumerate(image_list):
        logger.warning(f'{image_file=}')
        image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
        if image_file.name in image_data:
            frame_grid_strs = image_data[image_file.name]
        else:
            # 获取RGBA格式的边框像素
            edge_pixels_rgba = get_edge_pixels_rgba(image_raw)
            # 寻找主要颜色
            color_ratio, dominant_color = find_dominant_color(edge_pixels_rgba)
            # 获取颜色名称
            color_name = get_color_name(dominant_color)
            logger.info(
                f"边框颜色中出现次数最多的颜色：{dominant_color}, 颜色名称：{color_name}, {color_ratio=}")
            # 计算主要颜色遮罩
            frame_mask = compute_frame_mask(image_raw, dominant_color)
            ih, iw = frame_mask.shape[0:2]
            rec0 = (0, 0, iw, ih)
            pg = PicGrid(frame_mask, media_type)
            grids = []
            if pg.basic:
                rec0 = Rect(rec0, media_type)
                rec0.get_devide(frame_mask)
                grids.append(rec0)
            else:
                rects = pg.rects_dic[pg.judgement]
                for rect in rects:
                    get_frame_grid_recursive(rect, grids, frame_mask, media_type)
            frame_grid_strs = [rect.frame_grid_str for rect in grids]
            image_data[image_file.name] = frame_grid_strs

        pprint(frame_grid_strs)
        if len(frame_grid_strs) >= 2:
            # marked_frames = get_marked_frames(frame_grid_strs, image_raw)
            pass

    # Sort image_data dictionary by image file names
    image_data_sorted = {k: image_data[k] for k in natsorted(image_data)}

    # Save the image_data_sorted dictionary to the yml file
    with open(frame_yml_path, 'w') as yml_file:
        yaml.dump(image_data_sorted, yml_file)


@timer_decorator
@logger.catch
def step1_analyze_bubbles():
    if frame_yml_path.exists():
        with open(frame_yml_path, 'r') as yml_file:
            image_data = yaml.safe_load(yml_file)
    else:
        image_data = {}

    all_masks_old = get_valid_image_list(image_folder, mode='mask')

    for p, image_file in enumerate(image_list):
        logger.warning(f'{image_file=}')
        image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
        ih, iw = image_raw.shape[0:2]
        # ================矩形画格信息================
        if image_file.name in image_data:
            frame_grid_strs = image_data[image_file.name]
        else:
            frame_grid_strs = [f'0,0,{iw},{ih}~0,0,{iw},{ih}']
        # ================模型检测文字，文字显示为白色================
        if use_torch and comictextdetector_model is not None:
            comictextdetector_mask = get_comictextdetector_mask(image_raw)
        else:
            comictextdetector_mask = None
        # ================针对每一张图================
        for c in range(len(color_patterns)):
            # ================遍历每种气泡文字颜色组合================
            color_pattern = color_patterns[c]
            cp_bubble, cp_letter = color_pattern

            if isinstance(cp_bubble, list):
                # 气泡为渐变色
                color_bubble = ColorGradient(cp_bubble)
            else:
                # 气泡为单色
                color_bubble = Color(cp_bubble)

            if isinstance(cp_letter, list):
                # 文字为双色
                color_letter = ColorDouble(cp_letter)
                color_letter_big = color_letter
            else:
                # 文字为单色
                color_letter = Color(cp_letter)
                color_letter_big = Color(f'{cp_letter[:6]}-{max(color_letter.padding + 120, 180)}')

            if color_bubble.type == 'Color':
                bubble_mask = color_bubble.get_img(image_raw)
            else:  # ColorGradient
                img_tuple = color_bubble.get_img(image_raw)
                bubble_mask, left_sample, right_sample = img_tuple
            letter_mask = color_letter_big.get_img(image_raw)

            filtered_cnts = get_raw_bubbles(bubble_mask, letter_mask, comictextdetector_mask)
            # colorful_raw_bubbles = get_colorful_bubbles(image_raw, filtered_cnts)

            # ================切割相连气泡================
            single_cnts = seg_bubbles(filtered_cnts, bubble_mask, letter_mask, media_type)
            # ================通过画格重新排序气泡框架================
            single_cnts_recs = []
            for f in range(len(frame_grid_strs)):
                frame_grid_str = frame_grid_strs[f]
                # 使用正则表达式提取字符串中的所有整数
                int_values = list(map(int, findall(r'\d+', frame_grid_str)))
                # 按顺序分配值
                x, y, w, h, xx, yy, ww, hh = int_values
                single_cnts_rec = []
                for s in range(len(single_cnts)):
                    single_cnt = single_cnts[s]
                    if x <= single_cnt.cx <= x + w and y <= single_cnt.cy <= y + h:
                        single_cnts_rec.append(single_cnt)
                if media_type == 'Manga':
                    ax = -1
                else:
                    ax = 1
                single_cnts_rec.sort(key=lambda x: ax * x.cx + x.cy)
                single_cnts_recs.append(single_cnts_rec)
            single_cnts_ordered = list(chain(*single_cnts_recs))
            if len(single_cnts_ordered) >= 1:
                colorful_single_bubbles = get_colorful_bubbles(image_raw, single_cnts_ordered)

                alpha = 0.1
                # 创建一个带有10%透明度原图背景的图像
                transparent_image = zeros((image_raw.shape[0], image_raw.shape[1], 4), dtype=uint8)
                transparent_image[..., :3] = image_raw
                transparent_image[..., 3] = int(255 * alpha)
                # 在透明图像上绘制contours，每个contour使用不同颜色

                for s in range(len(single_cnts_ordered)):
                    bubble_cnt = single_cnts_ordered[s]
                    # 从 tab20 颜色映射中选择一个颜色
                    color = colormap_tab20(s % 20)[:3]
                    color_rgb = tuple(int(c * 255) for c in color)
                    color_bgra = color_rgb[::-1] + (255,)
                    drawContours(transparent_image, [bubble_cnt.contour], -1, color_bgra, -1)

                cp_preview_jpg = auto_subdir / f'{image_file.stem}-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.jpg'
                cp_mask_cnt_pic = auto_subdir / f'{image_file.stem}-Mask-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.png'
                write_pic(cp_preview_jpg, colorful_single_bubbles)
                write_pic(cp_mask_cnt_pic, transparent_image)

    # ================搬运气泡蒙版================
    auto_all_masks = get_valid_image_list(auto_subdir, mode='mask')
    # 如果步骤开始前没有气泡蒙版
    if not all_masks_old:
        for mask_src in auto_all_masks:
            mask_dst = image_folder / mask_src.name
            copy2(mask_src, mask_dst)


def get_single_cnts(image_file, image_raw, all_masks):
    ih, iw = image_raw.shape[0:2]
    black_background = zeros((ih, iw), dtype=uint8)

    mask_pics = [x for x in all_masks if x.stem.startswith(image_file.stem)]
    single_cnts = []
    # ================针对每一张图================
    for m in range(len(mask_pics)):
        mask_pic = mask_pics[m]
        logger.warning(f'{mask_pic=}')

        # ================获取轮廓================
        transparent_image = imdecode(fromfile(mask_pic, dtype=uint8), -1)
        # 将半透明的像素变为透明
        transparent_image[transparent_image[..., 3] < 255] = 0
        # 获取所有完全不透明的像素的颜色
        non_transparent_pixels = transparent_image[transparent_image[..., 3] == 255, :3]
        # 获取所有不透明像素的唯一颜色
        unique_colors = np.unique(non_transparent_pixels, axis=0)
        # 对于每种颜色，提取对应的contour
        contour_list = []
        for color in unique_colors:
            mask = np.all(transparent_image[..., :3] == color, axis=-1)
            mask = (mask & (transparent_image[..., 3] == 255)).astype(uint8) * 255
            contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            contour_list.extend(contours)

        # ================遍历每种气泡文字颜色组合================
        cp_str = mask_pic.stem.removeprefix(f'{image_file.stem}-Mask-')
        # ffffff-15-white~000000-60-black
        a_str, partition, b_str = cp_str.partition('~')
        a_parts = a_str.split('-')
        b_parts = b_str.split('-')
        color_pattern = [
            f'{a_parts[0]}-{a_parts[1]}',
            f'{b_parts[0]}-{b_parts[1]}',
        ]
        # print(f'{color_pattern=}')
        cp_bubble, cp_letter = color_pattern

        if isinstance(cp_bubble, list):
            # 气泡为渐变色
            color_bubble = ColorGradient(cp_bubble)
        else:
            # 气泡为单色
            color_bubble = Color(cp_bubble)

        if isinstance(cp_letter, list):
            # 文字为双色
            color_letter = ColorDouble(cp_letter)
            color_letter_big = color_letter
        else:
            # 文字为单
            color_letter = Color(cp_letter)
            color_letter_big = Color(f'{cp_letter[:6]}-{max(color_letter.padding + 120, 180)}')

        # ================获取对应的气泡文字蒙版================
        if color_bubble.type == 'Color':
            bubble_mask = color_bubble.get_img(image_raw)
        else:  # ColorGradient
            img_tuple = color_bubble.get_img(image_raw)
            bubble_mask, left_sample, right_sample = img_tuple
        letter_mask = color_letter_big.get_img(image_raw)

        for c in range(len(contour_list)):
            contour = contour_list[c]
            single_cnt = Contr(contour)

            # 在黑色背景上绘制单个轮廓的白色填充图像
            bit_white_bubble = drawContours(black_background.copy(), [single_cnt.contour], 0, 255, -1)

            # 获取原始图像在bit_white_bubble范围内的图像，其他部分为白色
            image_bubble_only = bitwise_and(image_raw, image_raw, mask=bit_white_bubble)
            image_bubble_only[bit_white_bubble == 0] = (255, 255, 255)

            # 通过将字符掩码与轮廓掩码相乘，得到只包含轮廓内部字符的图像
            letter_in_bubble = bitwise_and(letter_mask, letter_mask, mask=bit_white_bubble)
            # 获取所有非零像素的坐标
            px_pts = transpose(nonzero(letter_in_bubble))
            # 计算非零像素的数量
            px_area = px_pts.shape[0]
            # 获取所有非零像素的x和y坐标
            all_x = px_pts[:, 1]
            all_y = px_pts[:, 0]

            # 计算最小和最大的x、y坐标，并添加指定的padding
            min_x, max_x = np.min(all_x) - padding, np.max(all_x) + padding
            min_y, max_y = np.min(all_y) - padding, np.max(all_y) + padding

            # 限制padding后的坐标范围不超过原始图像的边界
            min_x, min_y = max(min_x, 0), max(min_y, 0)
            max_x, max_y = min(max_x, image_raw.shape[1]), min(max_y, image_raw.shape[0])

            # 从带有白色背景的图像中裁剪出带有padding的矩形区域
            if color_bubble.rgb_str == 'ffffff':
                cropped_image = image_bubble_only[min_y:max_y, min_x:max_x]
            else:
                cropped_image = letter_in_bubble[min_y:max_y, min_x:max_x]
            single_cnt.add_cropped_image(cropped_image)
            single_cnts.append(single_cnt)
    return single_cnts


def step2_OCR():
    if frame_yml_path.exists():
        with open(frame_yml_path, 'r') as yml_file:
            image_data = yaml.safe_load(yml_file)
    else:
        image_data = {}
    # ================气泡蒙版================
    all_masks = get_valid_image_list(image_folder, mode='mask')
    # ================DOCX文档================
    # 创建一个新的Document对象
    OCR_doc = Document()
    for i in range(len(image_list)):
        image_file = image_list[i]
        logger.warning(f'{image_file=}')

        OCR_doc.add_paragraph(image_file.stem)
        OCR_doc.add_paragraph('')

        image_raw = imdecode(fromfile(image_file, dtype=uint8), -1)
        ih, iw = image_raw.shape[0:2]

        # ================获取对应的文字图片================
        single_cnts = get_single_cnts(image_file, image_raw, all_masks)
        logger.debug(f'{len(single_cnts)=}')
        # ================矩形画格信息================
        if image_file.name in image_data:
            frame_grid_strs = image_data[image_file.name]
        else:
            frame_grid_strs = [f'0,0,{iw},{ih}~0,0,{iw},{ih}']
        # ================通过画格重新排序气泡框架================
        single_cnts_recs = []
        for f in range(len(frame_grid_strs)):
            frame_grid_str = frame_grid_strs[f]
            # 使用正则表达式提取字符串中的所有整数
            int_values = list(map(int, findall(r'\d+', frame_grid_str)))
            # 按顺序分配值
            x, y, w, h, xx, yy, ww, hh = int_values
            single_cnts_rec = []
            for s in range(len(single_cnts)):
                single_cnt = single_cnts[s]
                if x <= single_cnt.cx <= x + w and y <= single_cnt.cy <= y + h:
                    single_cnts_rec.append(single_cnt)
            if media_type == 'Manga':
                ax = -1
            else:
                ax = 1
            single_cnts_rec.sort(key=lambda x: ax * x.cx + x.cy)
            single_cnts_recs.append(single_cnts_rec)
        single_cnts_ordered = list(chain(*single_cnts_recs))

        for c in range(len(single_cnts_ordered)):
            single_cnt = single_cnts_ordered[c]
            # ================对裁剪后的图像进行文字识别================
            if SYSTEM in ['MAC', 'M1']:
                # ================Vision================
                recognized_text = ocr_by_vision(single_cnt.cropped_image, media_language)
                lines = []
                for text, (x, y, w, h), confidence in recognized_text:
                    # print(f"{text}[{confidence:.2f}] {x=:.2f}, {y=:.2f}, {w=:.2f}, {h=:.2f}")
                    lines.append(text)
            else:
                # ================tesseract================
                recognized_text = ocr_by_tesseract(single_cnt.cropped_image, media_language, vertical)
                # 将recognized_text分成多行
                lines = recognized_text.splitlines()
            # 去除空行
            non_empty_lines = [line for line in lines if line.strip()]
            # 将非空行合并为一个字符串
            cleaned_text = lf.join(non_empty_lines)
            print(cleaned_text)
            print('-' * 24)

            # ================将图片添加到docx文件中================
            cropped_image_pil = Image.fromarray(single_cnt.cropped_image)
            image_dpi = cropped_image_pil.info.get('dpi', (96, 96))[0]  # 获取图片的dpi，如果没有dpi信息，则使用默认值96
            with io.BytesIO() as temp_buffer:
                cropped_image_pil.save(temp_buffer, format=image_format.upper())
                temp_buffer.seek(0)
                pic_width_inches = cropped_image_pil.width / image_dpi
                OCR_doc.add_picture(temp_buffer, width=Inches(pic_width_inches))
            # ================将识别出的文字添加到docx文件中================
            OCR_doc.add_paragraph(cleaned_text)
            # 在图片和文字之间添加一个空行
            OCR_doc.add_paragraph('')

    # ================保存为DOCX================
    OCR_doc.save(OCR_docx_path)


def capitalize_sentence(sentence: str) -> str:
    """
    首字母大写处理句子，同时考虑缩写。

    :param sentence: 需要处理的句子。
    :return: 首字母大写处理后的句子。
    """
    abbreviations = ['Dr.', 'Ms.', 'Mr.', 'Mrs.']
    for abbr in abbreviations:
        sentence = sentence.replace(abbr, abbr.lower())
    capitalized_sentence = sentence[0].upper() + sentence[1:].lower()
    for abbr in abbreviations:
        capitalized_sentence = capitalized_sentence.replace(abbr.lower(), abbr)
    return capitalized_sentence


@logger.catch
@timer_decorator
def step3_translate():
    # 打开docx文件
    OCR_doc = Document(OCR_docx_path)

    # 读取并连接文档中的所有段落文本
    full_text = []
    for para in OCR_doc.paragraphs:
        full_text.append(para.text)
        # logger.debug(f'{para.text=}')

    index_dict = {}
    last_ind = 0
    inds = []
    for i in range(len(image_list)):
        image_file = image_list[i]
        logger.warning(f'{image_file=}')
        if image_file.stem in full_text[last_ind:]:
            ind = full_text[last_ind:].index(image_file.stem) + last_ind
            index_dict[image_file.stem] = ind
            inds.append(ind)
            last_ind = ind
            logger.debug(f'{ind=}')
    inds.append(len(full_text))
    # pprint(index_dict)

    pin = 0
    para_dict = {}
    for i in range(len(image_list)):
        image_file = image_list[i]
        if image_file.stem in index_dict:
            start_i = inds[pin] + 1
            end_i = inds[pin + 1]
            pin += 1
            para_dict[image_file.stem] = full_text[start_i:end_i]
    # pprint(para_dict)

    all_valid_paras = []
    for key in para_dict:
        paras = para_dict[key]
        valid_paras = [x for x in paras if x.strip() != '']
        if valid_paras:
            all_valid_paras.extend(valid_paras)

    simple_lines = []
    for a in range(len(all_valid_paras)):
        para = all_valid_paras[a]
        # 将多行文本替换为单行文本
        single_line_text = para.replace('\n', ' ')
        # 将文本分割成句子
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', single_line_text)
        # 对每个句子进行首字母大写处理，同时考虑缩写
        capitalized_sentences = [capitalize_sentence(sentence) for sentence in sentences if sentence]
        # 将处理后的句子连接成一个字符串
        processed_text = '. '.join(capitalized_sentences) + '.'
        simple_lines.append(processed_text)

    simple_text = lf.join(simple_lines)
    print(simple_text)

    chunks = []
    current_chunk = ""
    for line in simple_lines:
        # 检查将当前行添加到当前块后的长度是否超过最大字符数
        if len(current_chunk) + len(line) + 1 > max_chars:  # 加1是为了考虑换行符
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += line + "\n"

    # 添加最后一个块（如果有内容的话）
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    for chunk in chunks:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(chunk)
        print(translated_text)


@timer_decorator
def step4_lettering():
    pass


@timer_decorator
def generate_requirements(py_path, python_version):
    """
    生成给定Python文件中使用的非标准库的列表。

    :param py_path: 要分析的Python文件的路径。
    :type py_path: str
    :param python_version: Python版本的元组，默认为当前Python版本。
    :type python_version: tuple
    :return: 无
    """
    # 获取已安装的包及其版本
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    # 获取标准库模块列表
    stdlib_modules = set(stdlib_list(python_version))

    # 读取Python文件并解析语法树
    py_text = read_txt(py_path)
    root = parse(py_text)

    imports = []
    # 遍历语法树，提取import语句
    for node in walk(root):
        if isinstance(node, Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ImportFrom):
            if node.level == 0:
                imports.append(node.module)
    imported_modules = set(imports)
    requirements = []

    # 对于导入的每个模块，检查是否为非标准库模块
    for module in imported_modules:
        if module in installed_packages and module not in stdlib_modules:
            requirements.append(module)
    requirements.sort()
    requirements_text = lf.join(requirements)
    print(requirements_text)


def z():
    pass


if __name__ == "__main__":
    MomoHanhua = DOCUMENTS / '默墨汉化'
    Auto = MomoHanhua / 'Auto'
    Log = MomoHanhua / 'Log'
    ComicProcess = MomoHanhua / 'ComicProcess'
    MangaProcess = MomoHanhua / 'MangaProcess'
    ManhuaProcess = MomoHanhua / 'ManhuaProcess'

    MomoYolo = DOCUMENTS / '默墨智能'
    Storage = MomoYolo / 'Storage'

    make_dir(MomoHanhua)
    make_dir(Auto)
    make_dir(Log)
    make_dir(ComicProcess)
    make_dir(MangaProcess)
    make_dir(ManhuaProcess)

    make_dir(MomoYolo)
    make_dir(Storage)

    date_str = strftime('%Y_%m_%d')
    log_path = Log / f'日志-{date_str}.log'
    logger.add(
        log_path.as_posix(),
        rotation='500MB',
        encoding='utf-8',
        enqueue=True,
        compression='zip',
        retention='10 days',
        # backtrace=True,
        # diagnose=True,
        # colorize=True,
        # format="<green>{time}</green> <level>{message}</level>",
    )

    model_path = Storage / 'comictextdetector.pt.onnx'
    comictextdetector_model = None
    uoln = None

    # ================选择语言================
    # 不支持使用软件期间重选语言
    # 因为界面代码是手写的，没有 retranslateUi
    # lang_code = 'en_US'
    lang_code = 'zh_CN'
    # lang_code = 'zh_TW'
    # lang_code = 'ja_JP'
    qm_path = UserDataFolder / f'{APP_NAME}_{lang_code}.qm'

    color_patterns = [
        ['ffffff-15', '000000-60'],  # 白底黑字
    ]

    do_qt = False
    do_dev = False
    do_requirements = False


    def steps():
        pass


    do_mode = 'do_qt'  # 显示GUI
    # do_mode = 'do_dev'  # 开发调试
    # do_mode = 'do_requirements'  # 生成requirements

    if do_mode == 'do_qt':
        do_qt = True
    elif do_mode == 'do_dev':
        do_dev = True
    elif do_mode == 'do_requirements':
        do_requirements = True

    if do_qt:
        if model_path.exists():
            comictextdetector_model = dnn.readNetFromONNX(model_path.as_posix())
            uoln = comictextdetector_model.getUnconnectedOutLayersNames()

        appgui = QApplication(sys.argv)
        translator = QTranslator()
        translator.load(str(qm_path))
        QApplication.instance().installTranslator(translator)
        appgui.installTranslator(translator)
        main_qt()
    elif do_dev:
        # 给代码内容添加self.tr()
        # wrap_strings_with_tr()
        # 生成语言文件
        # self_translate_qt()

        image_folder = ComicProcess / 'Black Canary 001'
        frame_yml_path = image_folder.parent / f'{image_folder.name}.yml'
        OCR_docx_path = image_folder.parent / f'{image_folder.name}-OCR.docx'
        image_list = get_valid_image_list(image_folder)
        auto_subdir = Auto / image_folder.name
        make_dir(auto_subdir)

        media_type = 'Comic'
        media_language = 'English'
        target_language = 'zh-CN'
        vertical = False  # 设置为True以识别竖排文本
        if media_language == 'Japanese':
            vertical = True

        # step0_analyze_frames()
        # step1_analyze_bubbles()
        # step2_OCR()
        step3_translate()
    elif do_requirements:
        generate_requirements(py_path, python_version)
