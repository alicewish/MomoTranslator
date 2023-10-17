import codecs
import os
import os.path
import pickle
import re
import string
import sys
import webbrowser
from collections import Counter, OrderedDict
from colorsys import hsv_to_rgb, rgb_to_hsv
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from csv import reader, writer
from difflib import SequenceMatcher
from filecmp import cmp
from functools import wraps
from getpass import getuser
from hashlib import md5
from html import unescape
from io import BytesIO
from itertools import chain, zip_longest
from locale import getdefaultlocale
from math import cos, floor, radians, sin, sqrt
from operator import mod
from os.path import abspath, dirname, exists, expanduser, getmtime, getsize, isdir, isfile, normpath
from pathlib import Path
from platform import processor, system, uname
from pprint import pprint
from re import I, IGNORECASE, Pattern, escape, findall, finditer, match, search, sub
from shutil import copy2
from subprocess import PIPE, Popen, call
from time import sleep, strftime, time
from traceback import print_exc
from typing import Union
from uuid import getnode, uuid4
from warnings import filterwarnings

import numpy as np
import pyperclip
import spacy
import xmltodict
import yaml
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import fromarray
from PyQt6.QtCore import QPointF, QRectF, QSettings, QSize, QTranslator, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QBrush, QColor, QDoubleValidator, QFont, QFontMetrics, QIcon, QImage, \
    QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QDockWidget, QFileDialog, \
    QGraphicsDropShadowEffect, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, \
    QGraphicsPolygonItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListView, \
    QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QRadioButton, QStatusBar, \
    QTabWidget, QToolBar, QToolButton, QVBoxLayout, QWidget
from aip import AipOcr
from bs4 import BeautifulSoup, NavigableString, Tag
from cv2 import BORDER_CONSTANT, CHAIN_APPROX_SIMPLE, COLOR_BGR2BGRA, COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_BGRA2BGR, \
    COLOR_BGRA2RGBA, COLOR_GRAY2BGR, COLOR_GRAY2BGRA, COLOR_RGB2BGR, FILLED, INTER_LINEAR, RETR_EXTERNAL, RETR_LIST, \
    RETR_TREE, THRESH_BINARY, add, arcLength, bitwise_and, bitwise_not, \
    boundingRect, circle, contourArea, copyMakeBorder, cvtColor, dilate, dnn, drawContours, erode, findContours, \
    imdecode, imencode, inRange, line, mean, moments, pointPolygonTest, rectangle, resize, subtract, threshold
from deep_translator import GoogleTranslator
from docx import Document
from docx.shared import Inches
from easyocr import Reader
from fontTools.ttLib import TTCollection, TTFont
from loguru import logger
from mammoth import convert_to_html
from matplotlib import colormaps, font_manager
from natsort import natsorted
from nltk.corpus import names, words
from nltk.stem import WordNetLemmatizer
from numpy import arange, argmax, argwhere, array, asarray, ascontiguousarray, clip, diff, float32, fromfile, greater, \
    int32, maximum, mean, minimum, mod, ndarray, nonzero, ones, sqrt, squeeze, subtract, \
    transpose, uint8, unique, where, zeros, zeros_like
from paddleocr import PaddleOCR
from prettytable import PrettyTable
from psd_tools import PSDImage
from psutil import virtual_memory
from pypandoc import convert_text
from pytesseract import image_to_data, image_to_string
from qtawesome import icon as qicon
from ruamel.yaml import YAML
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points
from skimage.segmentation import watershed
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb, name_to_rgb, rgb_to_name

use_torch = True
# use_torch = False
if use_torch:
    import torch

filterwarnings("ignore", category=DeprecationWarning)

nlp = spacy.load("en_core_web_sm")
# python3 -m spacy info en_core_web_sm


# nltk.download('words')
# nltk.download('names')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# 合并常见单词和人名
good_words = set(words.words())
good_names = set(names.words())
good_words = good_words.union(good_names)

lemmatizer = WordNetLemmatizer()


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
homedir = Path(homedir)
DOWNLOADS = homedir / 'Downloads'
DOCUMENTS = homedir / 'Documents'

mac_address = ':'.join(findall('..', '%012x' % getnode()))
node_name = platform_uname.node

current_dir = dirname(abspath(__file__))
current_dir = Path(current_dir)

dirpath = os.getcwd()
ProgramFolder = Path(dirpath)
UserDataFolder = ProgramFolder / 'MomoHanhuaUserData'

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

APP_NAME = 'MomoTranslator'
MAJOR_VERSION = 2
MINOR_VERSION = 0
PATCH_VERSION = 0
APP_VERSION = f'v{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}'

APP_AUTHOR = '墨问非名'

if SYSTEM == 'WINDOWS':
    encoding = 'gbk'
    line_feed = '\n'
    cmct = 'ctrl'
else:
    encoding = 'utf-8'
    line_feed = '\n'
    cmct = 'command'

if SYSTEM in ['MAC', 'M1']:
    import applescript
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

ram = str(round(virtual_memory().total / (1024.0 ** 3)))

video_width = 1920
video_height = 1080
video_size = (video_width, video_height)

pylupdate = 'pylupdate6'
lrelease = 'lrelease'

window_title_prefix = f'{APP_NAME} {APP_VERSION}'

py_path = Path(__file__).resolve()
py_dev_path = py_path.parent / f'{py_path.stem}_dev.py'
py_i18n_path = py_path.parent / f'{py_path.stem}_i18n.py'
py_frag_path = py_path.parent / f'{py_path.stem}_frag.py'
py_lite_path = py_path.parent / f'{py_path.stem}_lite.py'
py_struct_path = py_path.parent / f'{py_path.stem}_struct.py'

max_chars = 5000

pictures_exclude = '加框,分框,框,涂白,填字,修图,-,copy,副本,拷贝,顺序,打码,测试,标注,边缘,标志,伪造'
pic_tuple = tuple(pictures_exclude.split(','))

docx_img_format = 'jpeg'
# docx_img_format = 'png'

spacing_ratio = 0.2

pos_0 = (0, 0)
pos_c = ('center', 'center')

color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_white = (255, 255, 255)
color_black = (0, 0, 0)

color_yellow = (0, 255, 255)  # 黄色
color_cyan = (255, 255, 0)  # 青色
color_magenta = (255, 0, 255)  # 洋红色
color_silver = (192, 192, 192)  # 银色
color_gray = (128, 128, 128)  # 灰色
color_maroon = (0, 0, 128)  # 褐红色
color_olive = (0, 128, 128)  # 橄榄色
color_purple = (128, 0, 128)  # 紫色
color_teal = (128, 128, 0)  # 蓝绿色
color_navy = (128, 0, 0)  # 海军蓝色
color_orange = (0, 165, 255)  # 橙色
color_pink = (203, 192, 255)  # 粉色
color_brown = (42, 42, 165)  # 棕色
color_gold = (0, 215, 255)  # 金色
color_lavender = (250, 230, 230)  # 薰衣草色
color_beige = (220, 245, 245)  # 米色
color_mint_green = (189, 255, 172)  # 薄荷绿
color_turquoise = (208, 224, 64)  # 绿松石色
color_indigo = (130, 0, 75)  # 靛蓝色
color_coral = (80, 127, 255)  # 珊瑚色
color_salmon = (114, 128, 250)  # 鲑鱼色
color_chocolate = (30, 105, 210)  # 巧克力色
color_tomato = (71, 99, 255)  # 番茄色
color_violet = (226, 43, 138)  # 紫罗兰色
color_goldenrod = (32, 165, 218)  # 金菊色
color_fuchsia = (255, 0, 255)  # 紫红色
color_crimson = (60, 20, 220)  # 深红色
color_dark_orchid = (204, 50, 153)  # 暗兰花色
color_slate_blue = (205, 90, 106)  # 石板蓝色
color_medium_sea_green = (113, 179, 60)  # 中等海洋绿色

rgba_white = (255, 255, 255, 255)
rgba_zero = (0, 0, 0, 0)
rgba_black = (0, 0, 0, 255)

outer_br_color = (255, 255, 0, 128)
inner_br_color = (0, 255, 255, 128)
index_color = (0, 0, 255, 255)

mark_px = [0, 0, 0, 1]

# 半透明紫色 (R, G, B, A)
semi_transparent_purple = (128, 0, 128, 168)

# rec_pad = 20
rec_pad = 10

pad = 5
lower_bound_white = array([255 - pad, 255 - pad, 255 - pad])
upper_bound_white = array([255, 255, 255])
lower_bound_black = array([0, 0, 0])
upper_bound_black = array([pad, pad, pad])

# 定义一个字符串，包含常见的结束标点符号，不能出现在断句最前
not_start_punct = ',，.。;；?？!！”’·-》>:】【]、)）…'
# 定义一个字符串，包含常见的起始标点符号，不能出现在断句最后
not_end_punct = '“‘《<【[(（'
# 定义一个字符串，包含常见的结束字，不能出现在断句最前
not_start_char = '上中下内出完的地得了么呢吗嘛呗吧着个就前世里图们来'
# 定义一个字符串，包含常见的起始字，不能出现在断句最后
not_end_char = '太每帮跟另向'

valid_chars = set('.,;!?-—`~!@#$%^&*()_+={}[]|\\:;"\'<>,.?/')

proper_nouns = {
    'Insomnia',
}

fine_words = [
    'jai',
    'jin',
    'jor',
    'fer',
    'binnu',
]

replacements = {
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
}

corrections = [
    ('LI', 'U'),
    ('X', 'Y'),
    # ('P', 'D'),
    # ('R', 'D'),
    ('ther', 'them'),
]

punct_rep_rules = {
    '.': ['…', '… ', '--'],
    '. ': ['…', '… ', '--'],
    ',': ['…', '.'],
    "'": ['“', '”', '"'],
    'I': [','],
    'l': ['j'],
    '‡': ['I'],
}

# 定义替换规则表
replace_rules = {
    '…': '...',
    'GET! T': 'GET IT',
    'WH!': 'NH!',
    'al/': 'all',
    '|': 'I',
    '/T': 'IT',
    '/': '!',
}

better_abbrs = {
    "YOL'RE": "YOU'RE",
    "YOL'LL": "YOU'LL",
    "YOL'VE": "YOU'VE",
    "WE'YE": "WE'VE",
    "WEIVE": "WE'VE",
    "IT'SA": "IT'S A",
    "IM": "I'M",
    "T'm": "I'M",
    "they'l": "they'll",
    "THEYIRE": "THEY'RE",
    "DONIT": "DON'T",
    "BOUT": "ABOUT",
    "SONLIVA": "SONUVA",
    "WORR'ED": "WORRIED",
    "SISTERIS": "SISTER'S",
    "LUCIFERIS": "LUCIFER'S",
    "CALLERIS": "CALLER'S",
    "IE": "IF",
    "15": "IS",
    "I6": "IS",
    "SPELIEL": "SPECIAL",
    "LACIES": "LADIES",
    "LBT": "LBJ",
    'ALOCA': 'A LOCA',
    # 'WH': 'NH',
    'YOW': 'YOU',
    'OKAX': 'OKAY',
    'TO-': 'TO--',
    'Daramit': 'Dammit',
}

# ================语气词================
interjections = ('huh', 'hn')

src_head_2c = ['Source', '目标语言']

media_type_dict = {
    0: 'Comic',
    1: 'Manga',
}

media_lang_dict = {
    0: 'English',
    1: 'Chinese Simplified',
    2: 'Chinese Traditional',
    3: 'Japanese',
    4: 'Korean',
}

easyocr_language_map = {
    'English': 'en'
}

paddleocr_language_map = {
    'English': 'en'
}

similar_chars_map = {
    'а': 'a',
    'А': 'A',
    'Ä': 'A',
    'д': 'A',
    'в': 'B',
    'В': 'B',
    'е': 'e',
    'Е': 'E',
    'и': 'u',
    'И': 'U',
    'к': 'k',
    'К': 'K',
    'н': 'H',
    'Н': 'H',
    'Ы': 'H',
    'М': 'M',
    'о': 'o',
    'О': 'O',
    'р': 'p',
    'Р': 'P',
    'с': 'c',
    'С': 'C',
    'т': 'T',
    'Т': 'T',
    'у': 'y',
    'У': 'Y',
    'х': 'x',
    'Х': 'X',
    'ч': '4',
    'Ч': '4',
    'ь': 'b',
    'Ь': 'B',
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

p_color = re.compile(r'([a-fA-F0-9]{6})-?(\d{0,3})', I)

# 使用matplotlib的tab20颜色映射
colormap_tab20 = colormaps['tab20']

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


# ================================基础函数区================================
def a2_base():
    return


def kernel(size):
    return ones((size, size), uint8)


def kernel_hw(h, w):
    return ones((h, w), uint8)


kernel1 = kernel(1)
kernel2 = kernel(2)
kernel3 = kernel(3)
kernel4 = kernel(4)
kernel5 = kernel(5)
kernel6 = kernel(6)
kernel7 = kernel(7)
kernel8 = kernel(8)
kernel9 = kernel(9)
kernel10 = kernel(10)
kernel12 = kernel(12)
kernel15 = kernel(15)
kernel20 = kernel(20)
kernel30 = kernel(30)
kernel40 = kernel(40)
kernel50 = kernel(50)
kernel60 = kernel(60)
kernalh1w5 = kernel_hw(1, 5)
kernalh1w6 = kernel_hw(1, 6)
kernalh1w7 = kernel_hw(1, 7)
kernalh1w8 = kernel_hw(1, 8)
kernalh1w9 = kernel_hw(1, 9)
kernalh1w10 = kernel_hw(1, 10)
kernalh1w11 = kernel_hw(1, 11)
kernalh1w12 = kernel_hw(1, 12)
kernalh1w13 = kernel_hw(1, 13)
kernalh1w14 = kernel_hw(1, 14)
kernalh1w15 = kernel_hw(1, 15)
kernalh1w16 = kernel_hw(1, 16)
kernalh1w17 = kernel_hw(1, 17)
kernalh1w18 = kernel_hw(1, 18)
kernalh1w19 = kernel_hw(1, 19)
kernalh1w20 = kernel_hw(1, 20)
kernalh5w1 = kernel_hw(5, 1)
kernalh8w1 = kernel_hw(8, 1)
kernalh10w1 = kernel_hw(10, 1)


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
    return bool(match(pattern, s))


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


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def reduce_list(input_list):
    try:
        # 尝试使用dict.fromkeys方法
        output_list = list(OrderedDict.fromkeys(input_list))
    except TypeError:
        # 如果发生TypeError（可能是因为列表包含不可哈希的对象），
        # 则改用更慢的方法
        output_list = []
        for input in input_list:
            if input not in output_list:
                output_list.append(input)
    return output_list


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


# @logger.catch
def get_valid_imgs(rootdir, mode='raw'):
    all_pics = get_files(rootdir, 'pic', True)
    jpgs = [x for x in all_pics if x.suffix in ('.jpg', '.jpeg')]
    pngs = [x for x in all_pics if x.suffix == '.png']

    all_masks = [x for x in pngs if '-Mask-' in x.stem]
    no_masks = [x for x in pngs if '-Mask-' not in x.stem]

    valid_jpgs = desel_list(jpgs, pic_tuple)
    valid_pngs = desel_list(no_masks, pic_tuple)

    valid_img_list = []
    if valid_jpgs:
        valid_img_list = valid_jpgs
    elif valid_pngs:
        valid_img_list = valid_pngs
    if mode == 'raw':
        return valid_img_list
    else:
        return all_masks


def iload_data(file_path, mode='yml'):
    if file_path.exists():
        with open(file_path, 'r' if mode == 'yml' else 'rb') as file:
            if mode == 'yml':
                return yaml.safe_load(file)
            elif mode == 'pkl':
                return pickle.load(file)
    return {}


# @logger.catch
def colored_text(text, color):
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'purple': '\033[95m',
        'highlight': '\033[91m',
        'end': '\033[0m'
    }
    return colors[color] + text + colors['end']


# @logger.catch
def simplify_areas(areas):
    # 简化Actual Areas的表示
    areas.sort()
    simplified = []
    start = areas[0]
    end = start
    for i in range(1, len(areas)):
        if areas[i] == end + 1:
            end = areas[i]
        else:
            if start == end:
                simplified.append(str(start))
            else:
                simplified.append(f"{start}~{end}")
            start = areas[i]
            end = start
    if start == end:
        simplified.append(str(start))
    else:
        simplified.append(f"{start}~{end}")
    return ', '.join(simplified)


# @logger.catch
def cal_words_data(raw_words_data):
    heights = [x[2] for x in raw_words_data]
    heights.sort()
    discard_count = int(0.1 * len(heights))
    lower_bound = heights[discard_count]
    upper_bound = heights[-discard_count - 1]
    words_data = [data for data in raw_words_data if lower_bound <= data[2] <= upper_bound]

    # 对words_data进行处理，把表示我的I替换成|
    for i, (word, word_format, height, area) in enumerate(words_data):
        # 使用正则表达式对line_word进行处理，匹配前面有字母或者后面有字母的"I"
        word = sub(r'(?<=[a-zA-Z])I|I(?=[a-zA-Z])', '|', word)
        words_data[i] = (word, word_format, height, area)

    # 获取所有独特的字符
    appeared_chars = set()
    unique_chars = set(''.join([word for word, word_format, height, black_px_area in words_data]))
    formatted_chars = set()
    for char in unique_chars:
        formatted_chars.add(char)  # 无格式字符
        formatted_chars.add(f"{char}_b")  # 粗体字符
        formatted_chars.add(f"{char}_i")  # 斜体字符
        formatted_chars.add(f"{char}_bi")  # 粗斜体字符
    char_to_index = {char: index for index, char in enumerate(formatted_chars)}

    # 初始化矩阵和向量
    A = zeros((len(words_data), len(formatted_chars)))
    b = zeros(len(words_data))
    weights = zeros(len(words_data))

    for i, (word, word_format, height, area) in enumerate(words_data):
        for char in word:
            char_key = char
            if word_format != '':
                char_key += f'_{word_format}'
            A[i, char_to_index[char_key]] += 1
            appeared_chars.add(char_key)
        b[i] = area
        weights[i] = len(word)

    # 使用加权最小二乘法解方程组
    W = np.diag(np.sqrt(weights))
    x, residuals, rank, s = np.linalg.lstsq(W @ A, W @ b, rcond=None)

    # x 是一个向量，其中每个元素是一个字符的期望黑色像素面积
    char_to_area = {char: x[char_to_index[char]] for char in formatted_chars}  # 使用formatted_chars

    # 输出结果
    char_table = PrettyTable()
    char_table.field_names = ["Character", "Expected Area", "Bold", "Italic", "Bold Italic"]
    for char in sorted(unique_chars):
        row = [
            char,
            round(char_to_area[char], 2) if char in appeared_chars else '',
            round(char_to_area[f"{char}_b"], 2) if f"{char}_b" in appeared_chars else '',
            round(char_to_area[f"{char}_i"], 2) if f"{char}_i" in appeared_chars else '',
            round(char_to_area[f"{char}_bi"], 2) if f"{char}_bi" in appeared_chars else ''
        ]
        char_table.add_row(row)
    if print_tables:
        print(char_table)

    word_table = PrettyTable()
    word_table.field_names = ['Word', 'Actual Area', 'Expected Area', 'Difference']
    for word, word_format, height, area in sorted(words_data, key=lambda x: x[0]):
        color_map = {
            '': word,
            'b': colored_text(word, 'blue'),
            'i': colored_text(word, 'green'),
            'bi': colored_text(word, 'purple')
        }
        colored_word = f"{color_map[word_format]}[{height}]"
        expect_area = sum(
            [char_to_area[char + ('' if word_format == '' else f'_{word_format}')] for char in word])
        difference = round(expect_area - area, 2)
        if abs(difference) >= 0.2 * area:
            difference = colored_text(str(difference), 'highlight')
        word_table.add_row([colored_word, area, round(expect_area, 2), difference])
    if print_tables:
        print(word_table)
    # 创建字典来存储字符和其期望的面积
    char_area_dict = {char: round(char_to_area[char], 2) for char in appeared_chars}

    # 创建字典来存储单词（包括格式后缀）和其期望的面积，以及另一个字典来存储单词的实际面积列表
    word_area_dict = {}
    word_actual_areas_dict = {}

    for word, word_format, height, actual_area in words_data:
        word_key = word + ('' if word_format == '' else f'_{word_format}')
        expect_area = sum([char_to_area[char + ('' if word_format == '' else f'_{word_format}')] for char in word])
        if word_key not in word_area_dict:
            word_area_dict[word_key] = [expect_area]
            word_actual_areas_dict[word_key] = [actual_area]
        else:
            word_area_dict[word_key].append(expect_area)
            word_actual_areas_dict[word_key].append(actual_area)

    # 计算字符出现次数
    char_count = {}
    for word, word_format, height, area in words_data:
        for char in word:
            char_key = char + ('' if word_format == '' else f'_{word_format}')
            char_count[char_key] = char_count.get(char_key, 0) + 1

    # 输出结果
    seg_char_table = PrettyTable()
    seg_char_table.field_names = ["#", "Character", "Expected Area"]
    for idx, (char, area) in enumerate(sorted(char_area_dict.items()), 1):
        count_suffix = f"[{char_count[char]}]" if char in char_count and char_count[char] > 1 else ""
        seg_char_table.add_row([idx, char, f"{area}{count_suffix}"])
    if print_tables:
        print(seg_char_table)
    seg_word_table = PrettyTable()
    seg_word_table.field_names = ['#', 'Word', 'Expected Area', 'Actual Areas']
    for idx, (word_key, areas) in enumerate(sorted(word_area_dict.items()), 1):
        avg_expect_area = round(sum(areas) / len(areas), 2)
        seg_word_table.add_row([idx, word_key, avg_expect_area, simplify_areas(word_actual_areas_dict[word_key])])
    if print_tables:
        print(seg_word_table)
    return char_area_dict, word_actual_areas_dict


@timer_decorator
# @logger.catch
def get_area_dic(area_yml):
    area_data = iload_data(area_yml)
    area_dic = {}
    for color_locate in area_data:
        if print_tables:
            print(color_locate)
        words_w_format_str = area_data[color_locate]
        words_w_format_str.sort()
        words_data = []
        for w in range(len(words_w_format_str)):
            word_w_format_str = words_w_format_str[w]
            parts = word_w_format_str.rsplit('|', 2)
            if len(parts) == 2:
                word = parts[0]
                word_format = ''
                nums_str = parts[1]
            elif len(parts) == 3:
                word = parts[0]
                word_format = parts[1]
                nums_str = parts[2]
            else:
                # 如果parts的长度大于3，那么word中可能包含'|'字符
                word = '|'.join(parts[:-2])
                word_format = parts[-2]
                nums_str = parts[-1]
            height, black_px_area = map(int, nums_str.split(','))
            word_data = (word, word_format, height, black_px_area)
            words_data.append(word_data)
        if len(words_data) >= 28:
            char_area_dict, word_area_dict = cal_words_data(words_data)
            area_dic[color_locate] = (char_area_dict, word_area_dict)
    return area_dic


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


def generate_md5(image_array):
    image_data = imencode('.png', image_array)[1].tostring()
    file_hash = md5()
    file_hash.update(image_data)
    return file_hash.hexdigest()


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


# @logger.catch
def write_pic(pic_path, picimg):
    pic_path = Path(pic_path)
    ext = pic_path.suffix
    temp_pic = pic_path.parent / f'{pic_path.stem}-temp{ext}'

    # 检查输入图像的类型
    if isinstance(picimg, bytes):
        # 如果是字节对象，直接写入文件
        with open(temp_pic, 'wb') as f:
            f.write(picimg)
    else:
        # 如果是PIL图像，转换为NumPy数组
        if isinstance(picimg, Image.Image):
            picimg = array(picimg)

            # 如果图像有三个维度，并且颜色为三通道，则进行颜色空间的转换
            if picimg.ndim == 3 and picimg.shape[2] == 3:
                picimg = cvtColor(picimg, COLOR_RGB2BGR)

        # 检查图像是否为空
        if picimg is None or picimg.size == 0:
            raise ValueError("The input image is empty.")

        # 保存临时图像
        imencode(ext, picimg)[1].tofile(temp_pic)

    # 检查临时图像和目标图像的md5哈希和大小是否相同
    if not pic_path.exists() or md5_w_size(temp_pic) != md5_w_size(pic_path):
        copy2(temp_pic, pic_path)

    # 删除临时图像
    if temp_pic.exists():
        os.remove(temp_pic)

    return pic_path


# #@logger.catch
def write_docx(docx_path, docu):
    temp_docx = docx_path.parent / 'temp.docx'
    if docx_path.exists():
        docu.save(temp_docx)
        if md5_w_size(temp_docx) != md5_w_size(docx_path):
            copy2(temp_docx, docx_path)
        if temp_docx.exists():
            os.remove(temp_docx)
    else:
        docu.save(docx_path)


def write_yml(yml_path, data):
    temp_yml = yml_path.parent / 'temp.yml'
    if yml_path.exists():
        with open(temp_yml, 'w', encoding='utf-8') as temp_file:
            yaml.dump(data, temp_file, default_flow_style=False, allow_unicode=True)
        if not cmp(temp_yml, yml_path):
            copy2(temp_yml, yml_path)
        if temp_yml.exists():
            os.remove(temp_yml)
    else:
        with open(yml_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)


def common_prefix(strings):
    """
    返回字符串列表中的共同前缀。

    :param strings: 字符串列表
    :return: 共同前缀
    """
    if not strings:
        return ""
    common_prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(common_prefix):
            common_prefix = common_prefix[:-1]
            if not common_prefix:
                return ""
    return common_prefix


def common_suffix(strings):
    """
    返回字符串列表中的共同后缀。

    :param strings: 字符串列表
    :return: 共同后缀
    """
    if not strings:
        return ""
    common_suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(common_suffix):
            common_suffix = common_suffix[1:]
            if not common_suffix:
                return ""
    return common_suffix


# @logger.catch
def parse_range(range_str):
    # 使用"~"分割字符串，并转化为浮点数列表
    # logger.warning(f'{range_str=}')
    range_strs = range_str.split('~')
    ranges = [float(x) for x in range_strs]
    return ranges


def lcs(X, Y):
    """
    计算两个字符串X和Y的最长公共子序列（Longest Common Subsequence, LCS）。

    :param X: 第一个需要比较的字符串。
    :param Y: 第二个需要比较的字符串。
    :return: 返回最长公共子序列。
    """
    m = len(X)  # X的长度
    n = len(Y)  # Y的长度

    # 初始化一个二维列表L，用于存储子问题的解
    L = [[0] * (n + 1) for i in range(m + 1)]

    # 动态规划填表
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # 从填好的表中构造LCS
    index = L[m][n]
    lcs = [''] * (index + 1)
    lcs[index] = ''
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(lcs)


def color_diff(str1, str2):
    """
    对比两个字符串，并用颜色标出第一个字符串中与第二个字符串不同的字符。

    :param str1: 第一个需要对比的字符串。
    :param str2: 第二个需要对比的字符串。
    :return: 返回第一个字符串，其中与第二个字符串不同的字符将被标红。
    """
    common = lcs(str1, str2)  # 计算最长公共子序列
    i = 0
    colored_str = ""
    for c in str1:
        if i < len(common) and c == common[i]:
            colored_str += c  # 相同字符不变
            i += 1
        else:
            colored_str += f"\033[91m{c}\033[0m"  # 不同字符标红

    return colored_str


def convert_str2num(s):
    if '.' in s:
        return float(s)
    else:
        return int(s)


# @logger.catch
def form2data(tess_zipped_data_form):
    tess_zipped_data = []
    for t in range(len(tess_zipped_data_form)):
        row = tess_zipped_data_form[t]
        if row and '|' in row:
            row_nums_str, par, row_text = row.partition('|')
            row_nums = row_nums_str.split(',')
            # 使用列表推导式将字符串转换为相应的整数或小数
            row_nums = [convert_str2num(x) for x in row_nums]
            tess_result = row_nums + [row_text]
            tess_zipped_data.append(tess_result)
        else:
            logger.error(f'[{t}]{row=}')
    return tess_zipped_data


def iread_csv(csv_file, pop_head=True, get_head=False):
    with open(csv_file, encoding='utf-8', mode='r') as f:
        f_csv = reader(f)
        if pop_head:
            # 获取首行并在需要时将其从数据中删除
            head = next(f_csv, [])
        else:
            head = []
        # 使用列表推导式简化数据读取
        idata = [tuple(row) for row in f_csv]
    if get_head:
        return idata, head
    else:
        return idata


# ================================基础图像函数区================================
def a3_pic():
    return


def rect2poly(x, y, w, h):
    # 四个顶点为：左上，左下，右下，右上
    points = [
        (x, y),  # 左上
        (x, y + h),  # 左下
        (x + w, y + h),  # 右下
        (x + w, y),  # 右上
    ]
    return points


def hex2int(hex_num):
    hex_num = f'0x{hex_num}'
    int_num = int(hex_num, 16)
    return int_num


def rgb2str(rgb_tuple):
    r, g, b = rgb_tuple
    color_str = f'{r:02x}{g:02x}{b:02x}'
    return color_str


def toBGR(image_raw):
    if len(image_raw.shape) == 2:
        image_raw = cvtColor(image_raw, COLOR_GRAY2BGR)
    elif image_raw.shape[2] == 3:
        pass
    else:
        image_raw = cvtColor(image_raw, COLOR_BGRA2BGR)
    return image_raw


def color2rgb(color):
    if isinstance(color, str):
        if color.startswith("#"):
            # 十六进制颜色
            return hex_to_rgb(color)
        else:
            # 颜色名称
            return name_to_rgb(color)
    elif isinstance(color, (tuple, list, ndarray)):
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


# @logger.catch
def get_color_name(color):
    rgb_color = color2rgb(color)
    try:
        color_name = rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color(rgb_color)
    return color_name


def print_colored_text(text, color_str):
    # 将16进制颜色转换为RGB
    rgb_color = color2rgb(color_str)
    r, g, b = rgb_color

    # 使用ANSI转义序列设置文本颜色
    ansi_color = f"\033[38;2;{r};{g};{b}m"
    reset_color = "\033[0m"

    # 打印彩色文本
    print(f"{ansi_color}{text}{reset_color}")


def highlight_diffs(content0, content):
    # 创建一个表格
    table = PrettyTable()
    table.field_names = ["Index", "Original Data", "Modified Data"]

    # 遍历content0和content的元素
    for idx, (c0, c) in enumerate(zip(content0, content), 0):
        # 如果元素不同，则添加高亮
        if c0 != c:
            table.add_row([idx, f"\033[1;31m{c0}\033[0m", f"\033[1;31m{c}\033[0m"])
        else:
            table.add_row([idx, c0, c])


def idx2label(idx):
    if 0 <= idx < 26:
        return string.ascii_uppercase[idx]
    elif 26 <= idx < 52:
        return string.ascii_lowercase[idx - 26]
    else:
        return str(idx)


def pt2tup(point):
    return (int(point.x), int(point.y))


def get_dist2rect(point, rect):
    x, y, w, h = rect
    dx = max(x - point.x, 0, point.x - (x + w))
    dy = max(y - point.y, 0, point.y - (y + h))
    return sqrt(dx ** 2 + dy ** 2)


def get_poly_by_cnt(contour_polys, cnt):
    # 这里假设每个 cnt 在 contour_polys 中只对应一个 poly
    for poly in contour_polys:
        if poly.cnt == cnt:
            return poly
    return None


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


def get_clipboard_data() -> Union[str, None]:
    """
    从剪贴板获取数据
    :return 如果没有错误，返回剪贴板数据；否则返回 None
    """
    process = Popen(['pbpaste'], stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    if not error:
        return output.decode('utf-8')
    else:
        print("Error:", error)
        return None


def get_ext(zipped, padding=15):
    lower = [min(x) for x in zipped]
    upper = [max(x) for x in zipped]
    ext_lower = [round(max(x - padding, 0)) for x in lower]
    ext_upper = [round(min(x + padding, 255)) for x in upper]
    ext_lower = tuple(reversed(ext_lower))
    ext_upper = tuple(reversed(ext_upper))
    ext = (ext_lower, ext_upper)
    return ext


def get_sample(video_img, left_rgb, right_rgb):
    zipped = list(zip(left_rgb, right_rgb))
    ext_lower, ext_upper = get_ext(zipped)
    logger.debug(f'{ext_lower=}, {ext_upper=}')
    sample_mask = inRange(video_img, array(ext_lower), array(ext_upper))
    return sample_mask


# ================================图像函数区================================
def a4_apple_script():
    return


def remove_common_indent(script: str) -> str:
    """
    删除脚本中每行开头的相同长度的多余空格
    :param script: 要处理的脚本
    :return: 删除多余空格后的脚本
    """
    lines = script.split('\n')
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return '\n'.join(line[min_indent:] for line in lines)


def run_apple_script(script: str) -> Union[str, None]:
    """
    执行 AppleScript 脚本
    :param script：要执行的 AppleScript 脚本
    :return 如果执行成功，返回执行结果；否则返回 None
    """
    if script:
        script = remove_common_indent(script)
        logger.debug(f'{script}')
        result = applescript.run(script)

        if result.code == 0:
            return result.out
        else:
            print(f'{result.err=}')


def get_browser_current_tab_url(browser: str) -> str:
    """
    获取浏览器当前标签页的 URL
    :param browser：浏览器名称，可以是 'Safari' 或 'Google Chrome'
    :return 当前标签页的 URL
    """
    if browser == 'Safari':
        apple_script = f'''
        tell application "{browser}"
            set current_url to URL of front document
            return current_url
        end tell
        '''
    elif browser == 'Google Chrome':
        apple_script = f'''
        tell application "{browser}"
            set current_url to URL of active tab of front window
            return current_url
        end tell
        '''
    else:
        print(f"Error: Unsupported browser {browser}.")
        return None
    return run_apple_script(apple_script)


def get_browser_current_tab_title(browser: str) -> str:
    """
    获取浏览器当前标签页的标题
    :param browser：浏览器名称，可以是 'Safari' 或 'Chrome'
    :return 当前标签页的标题
    """
    if browser == 'Safari':
        apple_script = f'''
        tell application "{browser}"
            set current_title to name of front document
            return current_title
        end tell
        '''
    elif browser == 'Google Chrome':
        apple_script = f'''
        tell application "{browser}"
            set current_title to title of active tab of front window
            return current_title
        end tell
        '''
    else:
        print(f"Error: Unsupported browser {browser}.")
        return None

    return run_apple_script(apple_script)


def get_browser_current_tab_html_fg(browser: str) -> str:
    """
    获取浏览器当前标签页的 HTML 内容
    :param browser：浏览器名称，可以是 'Safari' 或 'Chrome'
    :return 当前标签页的 HTML 内容
    """
    js_code = "document.documentElement.outerHTML;"

    if browser == 'Safari':
        apple_script = f'''
        tell application "{browser}"
            activate
            delay 1
            do JavaScript "{js_code}" in front document
            set the clipboard to result
            return (the clipboard as text)
        end tell
        '''
    elif browser == 'Google Chrome':
        apple_script = f'''
        tell application "{browser}"
            activate
            delay 1
            execute front window's active tab javascript "{js_code}"
            set the clipboard to result
            return (the clipboard as text)
        end tell
        '''
    else:
        print(f"Error: Unsupported browser {browser}.")
        return None

    return run_apple_script(apple_script)


def get_browser_current_tab_html(browser: str) -> str:
    """
    获取浏览器当前标签页的 HTML 内容
    :param browser：浏览器名称，可以是 'Safari' 或 'Chrome'
    :return 当前标签页的 HTML 内容
    """
    js_code = "document.documentElement.outerHTML;"

    if browser == 'Safari':
        apple_script = f'''
        tell application "{browser}"
            set curr_tab to current tab of front window
            do JavaScript "{js_code}" in curr_tab
            set the clipboard to result
            return (the clipboard as text)
        end tell
        '''
    elif browser == 'Google Chrome':
        apple_script = f'''
        tell application "{browser}"
            set curr_tab to active tab of front window
            execute curr_tab javascript "{js_code}"
            set the clipboard to result
            return (the clipboard as text)
        end tell
        '''
    else:
        print(f"Error: Unsupported browser {browser}.")
        return None

    return run_apple_script(apple_script)


def open_html_file_in_browser(file_path: str, browser: str) -> None:
    """
    在浏览器中打开 HTML 文件
    :param file_path：HTML 文件的路径
    :param browser：浏览器名称，可以是 'Safari' 或 'Chrome'
    """
    apple_script = f'''
    tell application "{browser}"
        activate
        open POSIX file "{file_path}"
    end tell
    '''
    return run_apple_script(apple_script)


@timer_decorator
def save_from_browser(browser: str):
    """
    主函数
    :param browser：浏览器名称，可以是 'Safari' 或 'Chrome'
    """
    current_url = get_browser_current_tab_url(browser)
    title = get_browser_current_tab_title(browser)
    logger.debug(f'{current_url=}')
    logger.warning(f'{title=}')
    if current_url and title:
        if current_url.startswith('https://chat.openai.com/'):
            chatgpt_html = ChatGPT / f'{title}-{Path(current_url).stem}.html'
            content = get_browser_current_tab_html(browser)
            # logger.info(f'{content}')
            soup = BeautifulSoup(content, 'html.parser')
            pretty_html = soup.prettify()
            write_txt(chatgpt_html, pretty_html)
    return chatgpt_html


# ================================图像函数区================================
def a5_frame():
    return


def get_edge_pxs(image_raw):
    h, w = image_raw.shape[:2]
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
    h, w = edge_pixels_rgba.shape[:2]
    frame_pxs = 2 * (h + w) * edge_size - 4 * edge_size ** 2
    color_ratio = counts[dominant_color_index] / frame_pxs
    dominant_color = colors[dominant_color_index]
    return color_ratio, dominant_color


def find_dominant_colors(edge_pixels_rgba, tolerance=10, top_colors=10):
    # 获取边框像素
    border_pixels = edge_pixels_rgba[where(edge_pixels_rgba[..., 3] != 0)]
    # 计算每种颜色的出现次数
    colors, counts = np.unique(border_pixels, axis=0, return_counts=True)

    h, w = edge_pixels_rgba.shape[:2]
    frame_pxs = 2 * (h + w) * edge_size - 4 * edge_size ** 2

    # 计算 color_ratios
    color_ratios = counts / frame_pxs

    # 创建包含颜色和 color_ratio 的元组列表
    color_and_ratios = sorted([(color, ratio) for color, ratio in zip(colors, color_ratios)], key=lambda x: -x[1])

    # 只保留前 10 种出现次数最多的颜色
    color_and_ratios = color_and_ratios[:top_colors]
    color_and_ratios_raw = deepcopy(color_and_ratios)

    combined_color_and_ratios = []
    while len(color_and_ratios) > 0:
        current_color, current_ratio = color_and_ratios.pop(0)
        combined_color_and_ratios.append((current_color, current_ratio))

        similar_color_indices = []
        for i, (color, ratio) in enumerate(color_and_ratios):
            if np.all(np.abs(current_color[:3] - color[:3]) <= tolerance):
                current_ratio += ratio
                similar_color_indices.append(i)

        # 移除已归类的相似颜色
        for index in sorted(similar_color_indices, reverse=True):
            color_and_ratios.pop(index)

    return color_and_ratios_raw, combined_color_and_ratios


def compute_frame_mask_single(image_raw, dominant_color, tolerance):
    dominant_color_int32 = array(dominant_color[:3]).astype(int32)
    lower_bound = maximum(0, dominant_color_int32 - tolerance)
    upper_bound = minimum(255, dominant_color_int32 + tolerance)
    mask = inRange(image_raw, lower_bound, upper_bound)
    return mask


def examine_white_line(row, min_dist, top_n, min_length):
    white_segments = np.split(row, where(diff(row) != 1)[0] + 1)
    sorted_segments = sorted(white_segments, key=len, reverse=True)

    top_segments = sorted_segments[:top_n]
    top_lengths = [len(segment) for segment in top_segments]

    valid_segments = []
    for idx, length in enumerate(top_lengths):
        if length >= min_length:
            valid_segments.append(top_segments[idx])

    if len(valid_segments) >= 2:
        distance = valid_segments[0][-1] - valid_segments[1][0]
        if distance >= min_dist:
            return sum(top_lengths[:2])
        else:
            return top_lengths[0]
    elif len(valid_segments) == 1:
        return top_lengths[0]
    else:
        return 0


def get_white_lines(binary_mask, method, media_type):
    """
    从给定的二值化图像中，提取白线。

    :param binary_mask: 二值化图像
    :param method: str，切分方法，可以是 'horizontal' 或 'vertical'
    :param media_type: str，媒体类型，例如 'Manga'

    :return: 经过校正的白线坐标列表。
    """
    grid_ratio = grid_ratio_dic.get(media_type)
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

    # ================细筛================
    # 对相邻白线分组
    cons = np.split(ori_gray, where(diff(ori_gray) != 1)[0] + 1)

    # 合并相邻的分组，如果它们之间的距离在 1～3 的范围内
    merged_cons = []
    cur_group = cons[0]

    for next_group in cons[1:]:
        if next_group[0] - cur_group[-1] <= 3:
            cur_group = np.concatenate((cur_group, next_group))
        else:
            merged_cons.append(cur_group)
            cur_group = next_group

    # 添加最后一个处理过的分组
    merged_cons.append(cur_group)

    # 转化为投影线段
    white_lines = [(x[0], x[-1] - x[0]) for x in merged_cons if len(x) >= 1]

    # ================最小厚度检验================
    white_lines_normal = [x for x in white_lines if x[1] > min_frame_length]
    return white_lines_normal


def get_recs(method, white_lines, iw, ih, offset_x, offset_y, media_type):
    """
    根据给定的方法和白色分割线，计算矩形区域列表。

    :param method: str，切分方法，可以是 'horizontal' 或 'vertical'
    :param white_lines: list of tuples，表示白色分割线的起点和长度，例如 [(start_pt1, length1), (start_pt2, length2), ...]
    :param iw: int，图像的宽度
    :param ih: int，图像的高度
    :param offset_x: int，横向偏移量
    :param offset_y: int，纵向偏移量
    :param media_type: str，媒体类型，例如 'Manga'

    :return: list of tuples，返回计算得到的矩形区域列表，每个矩形表示为 (x, y, w, h)
    """
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
        start_pt, leng = tu
        if edge_ratio * dep <= start_pt < start_pt + leng <= (1 - edge_ratio) * dep:
            true_dividers.append(start_pt)

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


# ================================qt函数区================================
def a6_pyqt():
    return


def copy2clipboard(text):
    clipboard = QApplication.clipboard()
    clipboard.setText(text)


def iact(text, icon=None, shortcut=None, checkable=False, toggled_func=None, trig=None):
    """创建并返回一个QAction对象"""
    action = QAction(text)
    if icon:  # 检查icon_name是否不为None
        action.setIcon(qicon(icon))
    if shortcut:
        action.setShortcut(QKeySequence(shortcut))
    if checkable:
        action.setCheckable(True)
    if toggled_func:
        action.toggled.connect(toggled_func)
    if trig:
        action.triggered.connect(trig)
    return action


def ibut(text, icon):
    button = QToolButton()
    button.setIcon(qicon(icon))
    button.setCheckable(True)
    button.setText(text)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    return button


def get_search_regex(search_text, case_sensitive, whole_word, use_regex) -> Pattern:
    """根据搜索条件，返回对应的正则表达式模式对象。"""

    # 如果不区分大小写，设置正则标志为忽略大小写
    flags = IGNORECASE if not case_sensitive else 0

    # 如果不使用正则表达式，对搜索文本进行转义，避免特殊字符影响匹配
    if not use_regex:
        search_text = escape(search_text)

    # 如果选择全词匹配，为搜索文本添加边界匹配符
    if whole_word:
        search_text = fr"\b{search_text}\b"

    # 尝试编译正则表达式，如果失败返回None
    try:
        return re.compile(search_text, flags)
    except re.error:
        return None


# ================================图像函数区================================
def a9_dev():
    return


def open_in_viewer(file_path):
    if sys.platform == 'win32':
        os.startfile(normpath(file_path))
    elif sys.platform == 'darwin':
        Popen(['open', file_path])
    else:
        Popen(['xdg-open', file_path])


def open_in_explorer(file_path):
    folder_path = dirname(file_path)
    if sys.platform == 'win32':
        Popen(f'explorer /select,"{normpath(file_path)}"')
    elif sys.platform == 'darwin':
        Popen(['open', '-R', file_path])
    else:
        Popen(['xdg-open', folder_path])


def open_in_ps(file_path):
    if sys.platform == 'win32':
        photoshop_executable_path = "C:/Program Files/Adobe/Adobe Photoshop CC 2019/Photoshop.exe"  # 请根据您的Photoshop安装路径进行修改
        Popen([photoshop_executable_path, file_path])
    elif sys.platform == 'darwin':
        photoshop_executable_path = "/Applications/Adobe Photoshop 2021/Adobe Photoshop 2021.app"  # 修改此行
        Popen(['open', '-a', photoshop_executable_path, file_path])
    else:
        logger.warning("This feature is not supported on this platform.")


# 处理单个字体
# @logger.catch
def process_font(font):
    name_table = font['name']
    # 中文名的langID
    zh_lang_id = 2052
    en_lang_id = 1033
    # nameID 4 对应的是完整的字体名称
    name_id = 4
    # 首先查找是否有中文名称
    font_name = name_table.getName(name_id, 3, 1, zh_lang_id)
    if font_name is None:
        # 如果没有中文名称，查找英文名称
        font_name = name_table.getName(name_id, 3, 1, en_lang_id)
    if font_name is None:
        return None
    return font_name.toUnicode()


# @logger.catch
def update_font_metadata(font_meta_csv, font_head):
    font_meta_list = []
    if font_meta_csv.exists():
        font_meta_list = iread_csv(font_meta_csv, True, False)
    else:
        for i in range(len(system_fonts)):
            font = system_fonts[i]
            font_path = Path(font)
            if font_path.exists():
                file_size = os.path.getsize(font) / 1024  # size in KB
                imes = f'Font {i + 1}: {font}, Size: {file_size:.2f} KB'
                # 读取字体文件
                meta_list = []
                if font_path.suffix.lower() == '.ttc':
                    font_collection = TTCollection(font_path)
                    # 对于TTC文件，处理每个字体
                    for font in font_collection.fonts:
                        font_name = process_font(font)
                        postscript_name = font['name'].getName(6, 3, 1, 0x409)
                        if postscript_name:
                            postscript_name = postscript_name.toStr()
                        meta = (font_name, postscript_name)
                        meta_list.append(meta)
                else:
                    font = TTFont(font_path)
                    # 对于TTF文件，处理单个字体
                    font_name = process_font(font)
                    postscript_name = font['name'].getName(6, 3, 1, 0x409)
                    if postscript_name:
                        postscript_name = postscript_name.toStr()
                    meta = (font_name, postscript_name)
                    meta_list.append(meta)
                meta_list = [x for x in meta_list if x[0] and x[1]]
                if meta_list:
                    imes += f', Meta: {meta_list=}'
                    meta_list0 = meta_list[0]
                    font_meta = (font_path.name,) + meta_list0
                    font_meta_list.append(font_meta)
                logger.debug(imes)
        font_meta_list = reduce_list(font_meta_list)
        font_meta_list.sort()
        write_csv(font_meta_csv, font_meta_list, font_head)
    return font_meta_list


def wrap_strings_with_tr(py_path):
    """
    该函数用于在 Python 文件中自动将从 QWidget 或 QMainWindow 继承的类中的字符串包装在 self.tr() 中。
    这有助于支持字符串的多语言翻译。

    :param py_path: 要处理的 Python 文件的路径。
    """

    # 从文件中读取内容
    content = read_txt(py_path)

    # 匹配从 QWidget 或 QMainWindow 继承的类
    class_pattern = r'class\s+\w+\s*\((?:\s*\w+\s*,\s*)*(QWidget|QMainWindow)(?:\s*,\s*\w+)*\s*\)\s*:'
    class_matches = finditer(class_pattern, content)

    # 在这些类中查找没有包裹在 self.tr() 中的字符串
    string_pattern = r'(?<!self\.tr\()\s*(".*?"|\'.*?\')(?!\s*\.(?:format|replace)|\s*\%|settings\.value)(?!\s*(?:setIcon|icon|setShortcut|QKeySequence|QSettings)\s*\()'

    # 特殊关键字，我们不希望替换包含这些关键字的字符串
    special_keywords = [...]  # 请在这里定义特殊关键字列表

    for class_match in class_matches:
        class_start = class_match.start()
        next_match = search(r'class\s+\w+\s*\(', content[class_start + 1:])
        if next_match:
            class_end = next_match.start()
        else:
            class_end = len(content)

        class_content = content[class_start:class_end]

        matches = finditer(string_pattern, class_content)
        new_class_cont = class_content
        for match in reversed(list(matches)):
            # 检查是否需要替换文本
            str_value = match.group(1)[1:-1]

            # 获取字符串所在的整行
            line_start = class_content.rfind('\n', 0, match.start()) + 1
            line_end = class_content.find('\n', match.end())
            line = class_content[line_start:line_end]

            # 检查字符串是否为十进制数或包含逗号，若是则跳过
            if is_decimal_or_comma(str_value) or str_value.endswith('%'):
                continue

            # 检查行中是否包含我们不需要替换的关键字
            if any(keyword in line for keyword in special_keywords):
                continue

            start = match.start(1)
            end = match.end(1)
            logger.info(f'正在修改: {match.group(1)}')
            # 使用单引号包裹字符串
            new_class_cont = new_class_cont[:start] + f'self.tr(\'{match.group(1)[1:-1]}\')' + new_class_cont[end:]
        content = content[:class_start] + new_class_cont + content[class_start + class_end:]
    updated_content = content

    print(updated_content)
    write_txt(py_i18n_path, updated_content)


@logger.catch
def self_translate_qt(py_path, language_tuples):
    """
    本函数用于提取可翻译字符串，生成 .ts 文件，更新翻译文件并将 .ts 文件转换为 .qm 文件。

    :param py_path: 一个包含 Python 文件的路径对象，用于从中提取可翻译字符串。
    :param language_tuples: 一个包含语言信息的元组列表，每个元组包括语言代码、英文名称、中文名称和自称名称。
    """
    for i in range(len(language_tuples)):
        language_tuple = language_tuples[i]
        lang_code, en_name, cn_name, self_name = language_tuple

        ts_file = UserDataFolder / f'{APP_NAME}_{lang_code}.ts'
        csv_file = UserDataFolder / f'{APP_NAME}_{lang_code}.csv'

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

        # 提取可翻译的字符串生成ts文件
        cmd = f'{pylupdate} {py_path.as_posix()} -ts {ts_file.as_posix()}'
        logger.debug(f'{cmd}')
        call(cmd, shell=True)
        # 解析 ts 文件
        xml_text = read_txt(ts_file)
        doc_parse = xmltodict.parse(xml_text)
        TS = doc_parse['TS']

        contexts = TS['context']
        updated_contexts = deepcopy(contexts)
        missing_sources = []

        for c in range(len(contexts)):
            context = contexts[c]
            updated_context = deepcopy(context)
            messages = context['message']
            updated_messages = deepcopy(messages)
            for m in range(len(updated_messages)):
                message = updated_messages[m]
                source = message['source']
                if source not in existing_sources:
                    logger.debug(f'{source=}')
                    missing_sources.append(source)
                if source in en2dst:
                    message['translation'] = en2dst[source]
            updated_context['message'] = updated_messages
            updated_contexts[c] = updated_context

        doc_parse['TS']['context'] = updated_contexts
        if missing_sources:
            new_src = deepcopy(src)
            for missing_source in missing_sources:
                new_src.append([missing_source, ''])
            write_csv(csv_file, new_src, src_head_2c)

        # 保存更新后的 ts 文件
        with ts_file.open('w', encoding='utf-8') as f:
            f.write(xmltodict.unparse(doc_parse))

        # 生成 qm
        cmd = f'{lrelease} {ts_file.as_posix()}'
        call(cmd, shell=True)


def get_tolerance(color_name0):
    tolerance = normal_tolerance
    if color_name0 == 'white':
        tolerance = white_tolerance
    elif color_name0 == 'black':
        tolerance = black_tolerance
    return tolerance


def get_CTD_mask(image):
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
    CTD_model.setInput(blob)
    blks, mask, lines_map = CTD_model.forward(uoln)
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


def get_filter_inds(good_inds, hierarchy):
    # 对有包含关系的轮廓，排除父轮廓
    filter_inds = []
    for i in range(len(good_inds)):
        contour_index = good_inds[i]
        parent_index = hierarchy[0][contour_index][3]
        is_parent_contour = False

        # 检查是否有任何 good_index 轮廓将此轮廓作为其父轮廓
        for other_index in good_inds:
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

        # 如果此轮廓不是任何 good_index 轮廓的父轮廓，将其添加到 filter_inds
        if not is_parent_contour:
            filter_inds.append(contour_index)
    return filter_inds


# @logger.catch
def get_colorful_bubbles(image_raw, bubble_cnts):
    # 将原始图像转换为PIL图像
    image_pil = fromarray(cvtColor(image_raw, COLOR_BGR2RGB))
    # 创建一个与原图大小相同的透明图像
    overlay = Image.new('RGBA', image_pil.size, rgba_zero)
    draw = ImageDraw.Draw(overlay)

    for f in range(len(bubble_cnts)):
        bubble_cnt = bubble_cnts[f]
        contour_points = [tuple(point[0]) for point in bubble_cnt.contour]
        # 如果坐标点数量少于2，那么跳过这个轮廓
        if len(contour_points) < 2:
            continue
        # 从 tab20 颜色映射中选择一个颜色
        color = colormap_tab20(f % 20)[:3]
        color_rgb = tuple(int(c * 255) for c in color)
        color_rgba = color_rgb + (int(255 * bubble_alpha),)
        draw.polygon(contour_points, fill=color_rgba)

    for f in range(len(bubble_cnts)):
        bubble_cnt = bubble_cnts[f]
        # 在轮廓中心位置添加序数
        text = str(f + 1)
        text_bbox = draw.textbbox(bubble_cnt.cxy, text, font=msyh_font100, anchor="mm")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        xy_pos = (bubble_cnt.cx - text_width // 2, bubble_cnt.cy - text_height // 2)
        draw.text(xy_pos, text, font=msyh_font100, fill=semi_transparent_purple)

    # 将带有半透明颜色轮廓的透明图像与原图混合
    image_pil.paste(overlay, mask=overlay)
    # 将 PIL 图像转换回 OpenCV 图像
    blended_img = cvtColor(array(image_pil), COLOR_RGB2BGR)
    return blended_img


# @logger.catch
def get_textblock_bubbles(image_raw, all_textblocks):
    ih, iw = image_raw.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)

    # 将原始图像转换为PIL图像
    image_pil = fromarray(cvtColor(image_raw, COLOR_BGR2RGB))
    # 创建一个与原图大小相同的透明图像
    overlay = Image.new('RGBA', image_pil.size, rgba_zero)
    draw = ImageDraw.Draw(overlay)

    all_core_brp_coords = []
    for b in range(len(all_textblocks)):
        textblock = all_textblocks[b]
        textlines = textblock.textlines
        textblock_colorf = colormap_tab20(b % 20)[:3]
        textblock_rgb = tuple(int(c * 255) for c in textblock_colorf)
        textblock_rgba = textblock_rgb + (int(255 * textblock_alpha),)
        # 将RGB颜色转换为HSV
        r, g, b = textblock_rgb
        h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        contour_points = [tuple(point[0]) for point in textblock.block_contour]
        if len(textlines) > 0:
            mask = drawContours(black_bg.copy(), [array(contour_points)], -1, 255, FILLED)
            dilated_mask = dilate(mask, kernel5, iterations=1)
            contours, _ = findContours(dilated_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            points = [tuple(point[0]) for point in contour]
            draw.polygon(points, fill=textblock_rgba)

            for l in range(len(textlines)):
                textline = textlines[l]
                textwords = textline.textwords
                # 对颜色进行微调以生成新的颜色
                textline_color = [(c + (l % 5) * 0.03) % 1.0 for c in textblock_colorf]
                # 将颜色值限定在0-255的范围内
                textline_rgb = tuple(min(max(int(c * 255), 0), 255) for c in textline_color)
                textline_rgba = textline_rgb + (int(255 * textline_alpha),)
                draw.rectangle(textline.br_uv, fill=textline_rgba)

                for w in range(len(textwords)):
                    textword = textwords[w]
                    # 改变色相值以得到不同的颜色
                    h = (h + (w % 5) * 0.2) % 1.0
                    textword_color = hsv_to_rgb(h, s, v)
                    textword_rgb = tuple(min(max(int(c * 255), 0), 255) for c in textword_color)
                    textword_rgba = textword_rgb + (int(255 * textword_alpha),)
                    draw.rectangle(textword.br_uv, fill=textword_rgba)

            contour_pairs = zip(contour_points, contour_points[1:] + [contour_points[0]])
            for start_pt, end_point in contour_pairs:
                draw.line([start_pt, end_point], fill=textblock_rgb, width=2)
            if len(textlines) == 1:
                core_brp_coords = list(textblock.core_brp.exterior.coords)
                all_core_brp_coords.append(core_brp_coords)
    for a in range(len(all_core_brp_coords)):
        core_brp_coords = all_core_brp_coords[a]
        draw.polygon(core_brp_coords, outline=color_navy)

    # 将带有半透明颜色轮廓的透明图像与原图混合
    image_pil.paste(overlay, mask=overlay)
    # 将 PIL 图像转换回 OpenCV 图像
    blended_img = cvtColor(array(image_pil), COLOR_RGB2BGR)
    return blended_img


# @logger.catch
def get_raw_bubbles(bubble_mask, letter_mask, left_sample, right_sample, CTD_mask):
    ih, iw = bubble_mask.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)
    all_contours, hierarchy = findContours(bubble_mask, RETR_TREE, CHAIN_APPROX_SIMPLE)

    all_cnts = []
    for a in range(len(all_contours)):
        contour = all_contours[a]
        cnt = Contr(contour)
        all_cnts.append(cnt)

    good_inds = []
    for a in range(len(all_cnts)):
        cnt = all_cnts[a]
        br_w_ratio = cnt.br_w / iw
        br_h_ratio = cnt.br_h / ih
        br_ratio = cnt.area / (cnt.br_w * cnt.br_h)
        portion_ratio = cnt.area / (iw * ih)
        # ================对轮廓数值进行初筛================
        condition_a1 = area_min <= cnt.area <= area_max
        condition_a2 = perimeter_min <= cnt.perimeter <= perimeter_max
        if cnt.area >= 500:
            # 狭长气泡
            condition_a3 = thickness_min - 2 <= cnt.thickness <= thickness_max
        else:
            condition_a3 = thickness_min <= cnt.thickness <= thickness_max
        condition_a4 = br_w_min <= cnt.br_w <= br_w_max
        condition_a5 = br_h_min <= cnt.br_h <= br_h_max
        condition_a6 = br_wh_min <= min(cnt.br_w, cnt.br_h) <= max(cnt.br_w, cnt.br_h) <= br_wh_max
        condition_a7 = br_wnh_min <= cnt.br_w + cnt.br_h <= br_wnh_max
        condition_a8 = br_w_ratio_min <= br_w_ratio <= br_w_ratio_max
        condition_a9 = br_h_ratio_min <= br_h_ratio <= br_h_ratio_max
        condition_a10 = br_ratio_min <= br_ratio <= br_ratio_max
        condition_a11 = portion_ratio_min <= portion_ratio <= portion_ratio_max
        condition_a12 = area_perimeter_ratio_min <= cnt.area_perimeter_ratio <= area_perimeter_ratio_max

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
            condition_a11,
            condition_a12,
        ]

        if all(condition_as):
            # ================结合文字图像进一步筛选================
            # 使用白色填充轮廓区域
            filled_contour = drawContours(black_bg.copy(), [cnt.contour], 0, 255, -1)

            # 使用位运算将 mask 和 filled_contour 相交，得到轮廓内的白色像素
            bubble_in_contour = bitwise_and(bubble_mask, filled_contour)
            letter_in_contour = bitwise_and(letter_mask, filled_contour)
            bubble_px = np.sum(bubble_in_contour == 255)
            letter_px = np.sum(letter_in_contour == 255)

            # ================检测文字轮廓================
            letter_contours, letter_hierarchy = findContours(letter_in_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            letter_cnts = []
            for q in range(len(letter_contours)):
                letter_contour = letter_contours[q]
                letter_cnt = Contr(letter_contour)
                letter_cnts.append(letter_cnt)
            dot_cnts = [x for x in letter_cnts if x.area <= 5]

            # ================对轮廓数值进行初筛================
            bubble_px_ratio = bubble_px / cnt.area  # 气泡像素占比
            letter_px_ratio = letter_px / cnt.area  # 文字像素占比
            BnL_px_ratio = bubble_px_ratio + letter_px_ratio

            # ================计算气泡缩小一圈（5像素）内文字像素数量================
            # 步骤1：使用erode函数缩小filled_contour，得到一个缩小的轮廓
            eroded_contour = erode(filled_contour, kernel5, iterations=1)
            # 步骤2：从原始filled_contour中减去eroded_contour，得到轮廓的边缘
            contour_edges = subtract(filled_contour, eroded_contour)
            # 步骤3：在letter_mask上统计边缘上的白色像素数量
            edge_pixels = where(contour_edges == 255)
            letter_inner_px_cnt = 0
            for y, x in zip(*edge_pixels):
                if letter_mask[y, x] == 255:
                    letter_inner_px_cnt += 1

            # ================计算气泡扩大一圈（5像素）内文字像素数量================
            # 步骤1：使用dilate函数扩大filled_contour，得到一个扩大的轮廓
            dilated_contour = dilate(filled_contour, kernel5, iterations=1)
            # 步骤2：从扩大的dilated_contour中减去原始filled_contour，得到轮廓的外缘
            contour_outer_edges = subtract(dilated_contour, filled_contour)
            # 步骤3：在letter_mask上统计外缘上的白色像素数量
            outer_edge_pixels = where(contour_outer_edges == 255)
            letter_outer_px_cnt = 0
            for y, x in zip(*outer_edge_pixels):
                if letter_mask[y, x] == 255:
                    letter_outer_px_cnt += 1
            border_thickness = letter_outer_px_cnt / cnt.perimeter
            # logger.debug(f'{border_thickness=:.4f}')

            # ================检测文字轮廓================
            letter_contours, letter_hierarchy = findContours(letter_in_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            letter_cnts = []
            for q in range(len(letter_contours)):
                letter_contour = letter_contours[q]
                letter_cnt = Contr(letter_contour)
                letter_cnts.append(letter_cnt)
            max_out_letter_cnts = [x for x in letter_cnts if x.br_h > max_font_size or x.br_w > 2 * max_font_size]
            # logger.warning(f'{max_out_letter_cnts=}')

            # ================对轮廓数值进行细筛================
            mask_px_ratio = 0.1
            if CTD_mask is not None:
                mask_in_contour = bitwise_and(CTD_mask, filled_contour)
                mask_px = np.sum(mask_in_contour >= 30)
                mask_px_ratio = mask_px / cnt.area  # 文字像素占比

            condition_b12 = True
            if left_sample is not None:
                # 使用filled_contour和left_sample的位运算，计算交集
                left_sample_in_contour = bitwise_and(left_sample, filled_contour)
                # 计算交集内的白色像素数量
                left_sample_px = np.sum(left_sample_in_contour == 255)
                # 计算filled_contour的面积
                filled_contour_area = np.sum(filled_contour == 255)
                # 检查left_sample在filled_contour内部的面积是否不小于filled_contour面积的30%
                condition_b12 = left_sample_px >= 0.3 * filled_contour_area

            condition_b1 = bubble_px_ratio_min <= bubble_px_ratio <= bubble_px_ratio_max
            condition_b2 = letter_px_ratio_min <= letter_px_ratio <= letter_px_ratio_max
            condition_b3 = bubble_px_min <= bubble_px <= bubble_px_max
            condition_b4 = letter_px_min <= letter_px <= letter_px_max
            condition_b5 = BnL_px_ratio_min <= BnL_px_ratio <= BnL_px_ratio_max
            condition_b6 = edge_px_count_min <= letter_inner_px_cnt <= edge_px_count_max
            condition_b7 = 1.5 <= border_thickness <= 5
            condition_b8 = (len(max_out_letter_cnts) == 0)
            condition_b9 = mask_px_ratio_min <= mask_px_ratio <= mask_px_ratio_max
            condition_b10 = len(dot_cnts) <= 40
            condition_b11 = letter_cnts_min <= len(letter_cnts) <= letter_cnts_max
            condition_bs = [
                condition_b1,
                condition_b2,
                condition_b3,
                condition_b4,
                condition_b5,
                # condition_b6,
                # condition_b7,
                # condition_b8,
                condition_b9,
                # condition_b10,
                condition_b11,
                # condition_b12,
            ]

            if all(condition_bs):
                good_inds.append(a)
                # logger.warning(f'{a=}')

    filter_inds = get_filter_inds(good_inds, hierarchy)

    filter_cnts = []
    for g in range(len(filter_inds)):
        filter_index = filter_inds[g]
        filter_cnt = all_cnts[filter_index]
        filter_cnts.append(filter_cnt)
        print(f'[{g + 1}]{filter_cnt.br=},{filter_cnt.area=:.0f},{filter_cnt.perimeter=:.2f}')

    return filter_cnts


def order2yaml(order_yml, ordered_cnts, custom_cnts, img_file):
    order_data = iload_data(order_yml)
    content0 = [cnt.cxy_str for cnt in ordered_cnts]
    content = [cnt.cxy_str for cnt in custom_cnts]
    highlight_diffs(content0, content)
    if content != content0:
        order_data[img_file.name] = content
        bubble_data_sorted = {k: order_data[k] for k in natsorted(order_data)}
        write_yml(order_yml, bubble_data_sorted)
        logger.debug(f"已保存到{order_yml}")


class AppConfig:
    # 设置默认配置文件路径和用户配置文件路径
    default_config_yml = UserDataFolder / f'{APP_NAME}_config.yml'
    user_config_yml = UserDataFolder / f'{APP_NAME}_{processor()}_{ram}GB_config.yml'
    master_config_yml = ProgramFolder / f'{APP_NAME}_master_config.yml'
    logger.debug(f'当前用户配置文件名为：{user_config_yml.name}')

    def __init__(self, config_file, config_data):
        # 初始化时设置配置文件路径，并加载配置数据
        self.config_file = Path(config_file)
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.config_data = config_data

    @classmethod
    def load(cls):
        # 加载配置文件。如果用户配置文件不存在，则复制默认配置文件
        config_file = cls.user_config_yml
        if not config_file.exists():
            copy2(cls.default_config_yml, config_file)

        # 使用 ruamel.yaml 来读取配置文件，因为它可以保留文件中的注释和顺序
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with config_file.open('r') as f:
            config_data = yaml.load(f)

        if cls.master_config_yml.exists():
            with open(cls.master_config_yml, mode='r', encoding='utf-8') as yf:
                master_cfg = yaml.load(yf)
            config_data.update(master_cfg)
        return cls(config_file, config_data)

    def get(self, key, default_value=None):
        # 获取指定配置项的值，如果没有找到，则返回默认值
        return self.config_data.get(key, default_value)

    def set(self, key, value):
        # 设置指定配置项的值，并保存到配置文件中
        self.config_data[key] = value
        self.save()

    def save(self):
        # 保存配置数据到配置文件
        try:
            with self.config_file.open('w') as f:
                self.yaml.dump(self.config_data, f)
        except IOError as e:
            logger.error(f'保存配置文件时出错：{e}')
        else:
            logger.info(f'成功保存配置文件到 {self.config_file}')


# 普通颜色与范围
class Color:

    def __init__(self, color_str):
        self.type = self.__class__.__name__
        self.color_str = color_str
        self.m_color = p_color.match(color_str)
        self.rgb_str = self.m_color.group(1)
        self.padding = self.m_color.group(2)

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

        # 计算亮度
        self.luminance = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b

        if self.padding == '':
            if self.black:
                self.padding = black_padding
            elif self.white:
                self.padding = white_padding
            elif self.luminance > 128:  # 浅色
                self.padding = light_padding
            elif self.luminance < 64:  # 深色
                self.padding = dark_padding
            else:
                self.padding = normal_padding
        else:
            self.padding = int(self.padding)

        self.ext_lower = [int(max(x - self.padding, 0)) for x in self.bgr]
        self.ext_upper = [int(min(x + self.padding, 255)) for x in self.bgr]

        self.color_name = get_color_name(self.rgb)

        if self.padding == 15:
            self.descripts = [self.rgb_str, self.color_name]
        else:
            self.descripts = [self.rgb_str, f'{self.padding}', self.color_name]
        self.descript = '-'.join(self.descripts)

    def put_padding(self, padding):
        self.padding = padding

    def get_range_img(self, task_img):
        frame_mask = inRange(task_img, array(self.ext_lower), array(self.ext_upper))
        return frame_mask

    def __str__(self):
        return f"{self.type}('{self.rgb_str}', '{self.color_name}', {self.padding})"

    def __repr__(self):
        return f'{self}'


# 渐变颜色与范围
class ColorGradient:

    def __init__(self, color_str_list):
        # 获取类的名字
        self.type = self.__class__.__name__
        # 获取输入的颜色列表
        self.color_str_list = color_str_list
        # 将输入的颜色列表连接成字符串
        self.color_str = '-'.join(color_str_list)

        # 获取输入的颜色列表中的两种颜色
        self.left_color_str, self.right_color_str = color_str_list
        # 将输入的颜色转换为Color对象
        self.left_color = Color(self.left_color_str)
        self.right_color = Color(self.right_color_str)

        # 获取Color对象的rgb字符串表示
        self.rgbs = [self.left_color.rgb_str, self.right_color.rgb_str]
        # 将Color对象的rgb字符串表示连接成一个字符串
        self.rgb_str = '-'.join(self.rgbs)

        # 获取Color对象的颜色名字
        self.color_name = f'{self.left_color.color_name}-{self.right_color.color_name}'

        # 获取Color对象的描述
        self.descripts = [self.left_color.descript, self.right_color.descript]
        # 将Color对象的描述连接成一个字符串
        self.descript = '-'.join(self.descripts)

        # 获取Color对象的padding属性的最大值
        self.padding = int(max(self.left_color.padding, self.right_color.padding))

        # 将两种颜色的rgb值压缩成一个列表
        self.zipped = list(zip(self.left_color.rgb, self.right_color.rgb))
        # 计算两种颜色的rgb值的平均值
        self.middle = tuple([int(mean(x)) for x in self.zipped])

        # 将中间颜色的rgb值转换为字符串
        self.middle_str = rgb2str(self.middle)
        # 判断中间颜色是否为白色
        self.white = (self.middle_str == 'ffffff')
        # 判断中间颜色是否为黑色
        self.black = (self.middle_str == '000000')

        # 获取中间颜色的rgb值
        self.r, self.g, self.b = self.middle
        # 设置颜色的透明度为255
        self.a = 255
        # 获取颜色的rgba值
        self.rgb = (self.r, self.g, self.b)
        self.bgr = (self.b, self.g, self.r)
        self.rgba = (self.r, self.g, self.b, self.a)
        self.bgra = (self.b, self.g, self.r, self.a)

        # 获取颜色的扩展下限和上限
        self.ext_lower, self.ext_upper = get_ext(self.zipped)
        # 记录颜色的扩展下限和上限
        logger.debug(f'{self.ext_lower=}, {self.ext_upper=}')

    def get_range_img(self, task_img):
        # 获取在颜色扩展范围内的图像
        frame_mask = inRange(task_img, array(self.ext_lower), array(self.ext_upper))
        # 获取在左边颜色和中间颜色之间的图像
        left_sample = get_sample(task_img, self.left_color.rgb, self.middle)
        # 获取在右边颜色和中间颜色之间的图像
        right_sample = get_sample(task_img, self.right_color.rgb, self.middle)
        # 将以上三种图像打包成元组
        img_tuple = (frame_mask, left_sample, right_sample)
        # 返回包含三种图像的元组
        return img_tuple

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

        self.rgbs = [self.left_color.rgb_str, self.right_color.rgb_str]
        self.rgb_str = '-'.join(self.rgbs)

        self.color_name = f'{self.left_color.color_name}-{self.right_color.color_name}'

        self.descripts = [self.left_color.descript, self.right_color.descript]
        self.descript = '-'.join(self.descripts)

        self.padding = int(max(self.left_color.padding, self.right_color.padding))

    def get_range_img(self, video_img):
        left_color_mask = self.left_color.get_range_img(video_img)
        right_color_mask = self.right_color.get_range_img(video_img)
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
        if len(contour) >= 3:
            self.polygon = Polygon(np.vstack((contour[:, 0, :], contour[0, 0, :])))
        else:
            self.polygon = None

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
        self.br_pts = rect2poly(*self.br)
        self.brp = Polygon(self.br_pts)

        self.M = moments(self.contour)
        # 这两行是计算质点坐标
        if self.M['m00'] != 0:
            self.cx = int(self.M['m10'] / self.M['m00'])
            self.cy = int(self.M['m01'] / self.M['m00'])
        else:
            self.cx, self.cy = 0, 0
        self.cxy = (self.cx, self.cy)
        self.cyx = (self.cy, self.cx)
        self.cxy_str = f'{self.cx},{self.cy}'
        self.cyx_str = f'{self.cy},{self.cx}'

    def add_cropped_img(self, cropped_img, letter_coors, color_pattern):
        self.cropped_img = cropped_img
        self.letter_coors = letter_coors
        self.center_pt = letter_coors[-1]
        self.color_pattern = color_pattern

    def get_core_br(self, black_bg):
        # ================获取不受气泡尖影响的外接矩形================
        binary = drawContours(black_bg.copy(), [self.contour], -1, 255, FILLED)
        if self.br_rt >= 0.7:
            # ================本来就接近圆形================
            self.core_br = self.br
            self.core_contour0 = self.contour
        else:
            if self.area >= 4000:
                kernel = kernel40
            elif self.area >= 3000:
                kernel = kernel30
            elif self.area >= 2000:
                kernel = kernel20
            else:
                kernel = kernel10
            erosion = erode(binary, kernel, iterations=1)
            dilation = dilate(erosion, kernel, iterations=1)
            self.core_bulk = bitwise_and(dilation, binary)
            self.core_contours, _ = findContours(self.core_bulk, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            if self.core_contours:
                # 根据新的轮廓找到外接矩形的左上角点
                self.core_contour0 = self.core_contours[0]
                self.core_br = boundingRect(self.core_contour0)
            else:
                core_preview_png = current_dir / 'core_preview.png'
                write_pic(core_preview_png, binary)
                self.core_br = self.br
                self.core_contour0 = self.contour
        self.core_br_x, self.core_br_y, self.core_br_w, self.core_br_h = self.core_br

    def is_inside(self, another_cnt):
        """检查当前轮廓是否在另一个轮廓内部"""
        # 使用轮廓的中心点来检查
        result = pointPolygonTest(another_cnt.contour, self.cxy, False)
        return result > 0  # 如果返回的距离为正，当前轮廓在另一个轮廓内部

    def __str__(self):
        return f"{self.type}({self.area:.1f}, {self.perimeter:.2f}){self.br}"

    def __repr__(self):
        return f'{self}'


class TextWord:
    def __init__(self, letter_cnt, media_type):
        self.type = self.__class__.__name__
        self.letter_cnts = []
        self.media_type = media_type
        self.word_ext_px = word_ext_px
        if isinstance(letter_cnt, list):
            self.letter_cnt = letter_cnt[0]
            self.letter_cnts = letter_cnt
        else:
            self.letter_cnt = letter_cnt
            self.letter_cnts.append(self.letter_cnt)
        self.letter_count = len(self.letter_cnts)
        self.calc_brs()

    def calc_brs(self):
        self.br_x = min(letter.br_x for letter in self.letter_cnts)
        self.br_u = max(letter.br_x + letter.br_w for letter in self.letter_cnts)
        self.br_y = min(letter.br_y for letter in self.letter_cnts)
        self.br_v = max(letter.br_y + letter.br_h for letter in self.letter_cnts)
        self.br_w = self.br_u - self.br_x + 1
        self.br_h = self.br_v - self.br_y + 1
        self.br_m = int(self.br_x + 0.5 * self.br_w)
        self.br_n = int(self.br_y + 0.5 * self.br_h)
        self.br_center = (self.br_m, self.br_n)
        self.br_center_pt = Point(self.br_m, self.br_n)
        self.br_area = self.br_w * self.br_h
        self.br = (self.br_x, self.br_y, self.br_w, self.br_h)
        self.br_uv = (self.br_x, self.br_y, self.br_u, self.br_v)
        self.left_top = (self.br_x, self.br_y)
        self.left_bottom = (self.br_x, self.br_v)
        self.right_top = (self.br_u, self.br_y)
        self.right_bottom = (self.br_u, self.br_v)
        self.br_pts = rect2poly(*self.br)
        self.brp = Polygon(self.br_pts)
        # ================往右端略微延长================
        self.ext_br = (self.br_x, self.br_y, self.br_w + self.word_ext_px, self.br_h)
        self.ext_br_pts = rect2poly(*self.ext_br)
        self.ext_brp = Polygon(self.ext_br_pts)

    def add_letter_cnt(self, letter_cnt):
        self.letter_cnts.append(letter_cnt)
        self.letter_cnts.sort(key=lambda x: x.cx + x.cy)
        self.letter_count = len(self.letter_cnts)
        self.calc_brs()

    def add_letter_mask(self, letter_mask):
        self.letter_mask = letter_mask
        self.ih, self.iw = self.letter_mask.shape[0:2]
        self.black_bg = zeros((self.ih, self.iw), dtype=uint8)
        self.all_contours = [x.contour for x in self.letter_cnts]
        self.all_letters = drawContours(self.black_bg.copy(), self.all_contours, -1, 255, FILLED)
        self.letter_in_word = bitwise_and(self.letter_mask, self.all_letters)
        # 获取所有非零像素的坐标
        self.px_pts = transpose(nonzero(self.letter_in_word))
        # 计算非零像素的数量
        self.px_area = self.px_pts.shape[0]
        self.letter_area = self.px_area / len(self.letter_cnts)

    def __str__(self):
        return f"{self.type}[{len(self.letter_cnts)}]br{self.br}center{self.br_center}"

    def __repr__(self):
        return f'{self}'


class TextLine:

    def __init__(self, textword, media_type):
        self.type = self.__class__.__name__
        self.textwords = []
        self.textword = textword
        self.media_type = media_type
        self.line_ext_px = line_ext_px
        self.textwords.append(self.textword)
        self.letter_count = sum([word.letter_count for word in self.textwords])
        self.word_count = len(self.textwords)
        self.calc_brs()

    def calc_brs(self):
        self.br_x = min(word.br_x for word in self.textwords)
        self.br_u = max(word.br_x + word.br_w for word in self.textwords)
        self.br_y = min(word.br_y for word in self.textwords)
        self.br_v = max(word.br_y + word.br_h for word in self.textwords)
        self.br_w = self.br_u - self.br_x + 1
        self.br_h = self.br_v - self.br_y + 1
        # ================处理超大字号================
        if adapt_big_letter and self.br_h >= 2 * line_ext_px:
            self.line_ext_px = max(self.line_ext_px, int(0.8 * self.br_h))
            logger.warning(f'{self.br_h=}, {self.line_ext_px=}')
        self.br_m = int(self.br_x + 0.5 * self.br_w)
        self.br_n = int(self.br_y + 0.5 * self.br_h)
        self.br_center = (self.br_m, self.br_n)
        self.br_center_pt = Point(self.br_m, self.br_n)
        self.br_area = self.br_w * self.br_h
        self.br = (self.br_x, self.br_y, self.br_w, self.br_h)
        self.br_uv = (self.br_x, self.br_y, self.br_u, self.br_v)
        self.left_top = (self.br_x, self.br_y)
        self.left_bottom = (self.br_x, self.br_v)
        self.right_top = (self.br_u, self.br_y)
        self.right_bottom = (self.br_u, self.br_v)
        self.br_pts = rect2poly(*self.br)
        self.brp = Polygon(self.br_pts)
        # ================往右端略微延长================
        self.ext_br = (self.br_x, self.br_y, self.br_w + self.line_ext_px, self.br_h)
        self.ext_br_pts = rect2poly(*self.ext_br)
        self.ext_brp = Polygon(self.ext_br_pts)

    def add_textword(self, textword):
        self.textwords.append(textword)
        self.textwords.sort(key=lambda x: x.br_x + x.br_y)
        self.letter_count += textword.letter_count
        self.word_count = len(self.textwords)
        self.calc_brs()

    def __str__(self):
        return f"{self.type}[{len(self.textwords)}]br{self.br}center{self.br_center}"

    def __repr__(self):
        return f'{self}'


class iTextBlock:

    def __init__(self, textline, media_type):
        self.type = self.__class__.__name__
        self.textlines = []
        self.textline = textline
        self.media_type = media_type
        self.block_ext_px = block_ext_px
        self.textlines.append(self.textline)
        self.letter_count = sum([line.letter_count for line in self.textlines])
        self.word_count = sum([line.word_count for line in self.textlines])
        self.line_count = len(self.textlines)
        self.calc_brs()

    def calc_brs(self):
        self.br_x = min(textline.br_x for textline in self.textlines)
        self.br_u = max(textline.br_u for textline in self.textlines)
        self.br_y = min(textline.br_y for textline in self.textlines)
        self.br_v = max(textline.br_v for textline in self.textlines)
        self.br_w = self.br_u - self.br_x + 1
        self.br_h = self.br_v - self.br_y + 1
        self.br_m = int(self.br_x + 0.5 * self.br_w)
        self.br_n = int(self.br_y + 0.5 * self.br_h)
        self.br_center = (self.br_m, self.br_n)
        self.br_center_pt = Point(self.br_m, self.br_n)
        self.br_area = self.br_w * self.br_h
        self.br = (self.br_x, self.br_y, self.br_w, self.br_h)
        self.br_uv = (self.br_x, self.br_y, self.br_u, self.br_v)
        self.br_pts = rect2poly(*self.br)
        self.brp = Polygon(self.br_pts)

        # ================柱钩================
        self.core_br = (
            self.br_m - block_ratio * self.block_ext_px,
            self.br_y - self.block_ext_px,
            2 * block_ratio * self.block_ext_px,
            self.block_ext_px)
        self.core_br_pts = rect2poly(*self.core_br)
        self.core_brp = Polygon(self.core_br_pts)

        # ================包络多边形================
        # 根据文本块的所有文本行的外接矩形生成最小包络多边形
        # 先上后下，先左后右的顺序收集所有的外接矩形点
        all_rect_pts = []
        for textline in self.textlines:
            all_rect_pts.append((textline.left_top))  # 左上
            all_rect_pts.append((textline.left_bottom))  # 左下
        for textline in reversed(self.textlines):
            all_rect_pts.append((textline.right_bottom))  # 右下
            all_rect_pts.append((textline.right_top))  # 右上
        # 利用所有的外接矩形点构建包络多边形
        self.block_poly = Polygon(all_rect_pts)
        # 直接将这些点转化为OpenCV可以使用的轮廓格式，即(N,1,2)的NumPy数组
        self.block_contour = array(all_rect_pts).reshape((-1, 1, 2)).astype(int32)
        self.block_cnt = Contr(self.block_contour)
        x, y, w, h = boundingRect(self.block_contour)
        # 扩展矩形的四个边界
        x -= rec_pad
        y -= rec_pad
        w += 2 * rec_pad
        h += 2 * rec_pad
        # 从扩展后的矩形获取新的轮廓
        self.expanded_contour = array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]).reshape((-1, 1, 2)).astype(int32)
        self.expanded_cnt = Contr(self.expanded_contour)

    def add_textline(self, textline):
        self.textlines.append(textline)
        self.textlines.sort(key=lambda x: x.br_y)
        # ================处理超大字号================
        if textline.line_ext_px >= 50:
            self.block_ext_px = max(self.block_ext_px, textline.line_ext_px)
        self.letter_count += textline.letter_count
        self.word_count += textline.word_count
        self.line_count = len(self.textlines)
        self.calc_brs()

    def __str__(self):
        return f"{self.type}[{len(self.textlines)}]br{self.br}center{self.br_center}"

    def __repr__(self):
        return f'{self}'


class Rect:

    def __init__(self, rect_tup, media_type):
        self.type = self.__class__.__name__  # 获取类名作为类型
        self.rect_tup = rect_tup  # 矩形四元组（x, y, 宽, 高）
        self.media_type = media_type  # 媒体类型

        self.x, self.y, self.w, self.h = self.rect_tup  # 解构四元组到x, y, 宽, 高
        self.xx, self.yy, self.ww, self.hh = self.x, self.y, self.w, self.h
        self.rect_inner_tuple = (self.xx, self.yy, self.ww, self.hh)  # 内部矩形四元组

        self.x_max = self.x + self.w  # 矩形最大x值
        self.y_max = self.y + self.h  # 矩形最大y值

        self.area = self.w * self.h  # 矩形面积
        self.basic = False  # 是否为不可分割的基本方块，初始化为False

    def put_frame_mask(self, frame_mask_group):
        # 根据矩形位置和大小从原图中切割子图
        self.sub_frame_mask_group = []
        for frame_mask in frame_mask_group:
            self.sub_frame_mask = frame_mask[self.y:self.y_max, self.x:self.x_max]
            self.sub_frame_mask_group.append(self.sub_frame_mask)
        # 创建PicGrid对象
        self.pg = PicGrid(
            self.sub_frame_mask_group,  # 子图
            self.media_type,  # 媒体类型
            self.x,  # 矩形x值
            self.y,  # 矩形y值
        )
        self.get_rect_inner()

    def get_rect_inner(self):
        for sub_frame_mask in self.sub_frame_mask_group:
            self.sub_frame_mask = sub_frame_mask
            # 复制子图像
            self.cleaned_mask = self.sub_frame_mask.copy()

            # 在四条边上添加用户指定宽度的白色边框
            border_width = 3
            # 定义一个字典，用于映射方向和对应的数组切片
            direction_slices = {
                'top': np.s_[:border_width, :],
                'bottom': np.s_[-border_width:, :],
                'left': np.s_[:, :border_width],
                'right': np.s_[:, -border_width:]
            }

            # 遍历四个方向，检查黑色像素数量并在满足条件时添加边框
            for direction, slices in direction_slices.items():
                # 计算指定方向的黑色像素数量
                black_pixels = np.sum(self.cleaned_mask[slices] == 0)
                # 如果黑色像素数量小于等于100，则在该方向添加边框
                if black_pixels <= 100:
                    self.cleaned_mask[slices] = 255

            # 使用形态学操作移除噪声
            # self.cleaned_mask = morphologyEx(self.cleaned_mask, MORPH_OPEN, kernel15)
            # 进行膨胀操作
            self.cleaned_mask = dilate(self.cleaned_mask, kernel15, iterations=1)
            # 进行腐蚀操作
            self.cleaned_mask = erode(self.cleaned_mask, kernel15, iterations=1)

            # 找到子图中所有黑色像素的坐标
            black_pixels = where(self.cleaned_mask == 0)
            black_y, black_x = black_pixels

            if not black_x.size or not black_y.size:
                # 如果没有黑色像素，那么内部矩形就是整个矩形
                pass
            else:
                # 计算所有黑色像素的外接矩形
                min_x, max_x = np.min(black_x), np.max(black_x)
                min_y, max_y = np.min(black_y), np.max(black_y)

                # 更新内部矩形的坐标和大小
                self.xx = self.x + min_x
                self.yy = self.y + min_y
                self.ww = max_x - min_x + 1
                self.hh = max_y - min_y + 1

            self.rect_inner_tuple = (self.xx, self.yy, self.ww, self.hh)  # 内部矩形四元组
            self.frame_grid = [self.rect_tup, self.rect_inner_tuple]  # 矩形网格，包括原矩形和内部矩形
            self.frame_grid_str = '~'.join([','.join([f'{y}' for y in x]) for x in self.frame_grid])  # 矩形网格字符串表示
            if self.rect_inner_tuple != self.rect_tup:
                break

    def __str__(self):
        return f"{self.type}({self.x}, {self.y}, {self.w}, {self.h})"

    def __repr__(self):
        return f'{self}'


class PicGrid:

    def __init__(self, frame_mask_group, media_type, offset_x=0, offset_y=0):
        self.type = self.__class__.__name__  # 类型名称
        self.media_type = media_type  # 媒体类型
        self.offset_x = offset_x  # 水平偏移量
        self.offset_y = offset_y  # 垂直偏移量
        self.frame_mask_group = frame_mask_group
        self.frame_mask_ind = 0
        self.find_recs(self.frame_mask_ind)

    def find_recs(self, frame_mask_ind):
        self.frame_mask = self.frame_mask_group[frame_mask_ind]  # 二值化的掩码图
        self.basic = False  # 是否是基本单元格
        self.judgement = None  # 判断结果

        # if do_dev_pic:
        #     frame_mask_grid_png = current_dir / 'frame_mask_grid.png'
        #     write_pic(frame_mask_grid_png, self.frame_mask)

        # 图片长宽
        self.ih, self.iw = self.frame_mask.shape[0:2]  # 图片高度和宽度
        self.px_pts = transpose(nonzero(self.frame_mask))  # 非零像素点的坐标
        self.px_area = self.px_pts.shape[0]  # 非零像素点的数量

        # 计算白色像素的比例
        total_pixels = self.ih * self.iw  # 图片的总像素数量
        white_ratio = self.px_area / total_pixels  # 白色像素的比例

        # 水平和垂直白线
        self.white_lines_horizontal = get_white_lines(self.frame_mask, 'horizontal', media_type)  # 水平白线
        self.white_lines_vertical = get_white_lines(self.frame_mask, 'vertical', media_type)  # 垂直白线
        self.white_lines = {
            'horizontal': self.white_lines_horizontal,  # 水平白线
            'vertical': self.white_lines_vertical  # 垂直白线
        }

        # 水平和垂直矩形
        self.rects_horizontal = get_recs(
            'horizontal',
            self.white_lines_horizontal,
            self.iw,
            self.ih,
            self.offset_x,
            self.offset_y,
            self.media_type,
        )  # 水平矩形
        self.rects_vertical = get_recs(
            'vertical',
            self.white_lines_vertical,
            self.iw,
            self.ih,
            self.offset_x,
            self.offset_y,
            self.media_type,
        )  # 垂直矩形
        self.rects_dic = {
            'horizontal': self.rects_horizontal,  # 水平矩形
            'vertical': self.rects_vertical  # 垂直矩形
        }

        # 判断矩形排列方向
        self.rects = None
        if len(self.rects_horizontal) == len(self.rects_vertical) == 1 or white_ratio >= max_white_ratio:
            self.basic = True  # 如果只有一个矩形，设为基本单元格
        elif len(self.rects_horizontal) >= 2:
            # 如果水平矩形数量大于等于2，判断结果为'horizontal'
            self.judgement = 'horizontal'
        else:
            # 否则判断结果为'vertical'
            self.judgement = 'vertical'
        if self.judgement:
            self.rects = self.rects_dic[self.judgement]  # 根据判断结果，选择对应的矩形

        # 检查self.basic和self.frame_mask_ind
        if self.basic and self.frame_mask_ind < len(self.frame_mask_group) - 1:
            self.frame_mask_ind += 1  # 增加self.frame_mask_ind
            self.find_recs(self.frame_mask_ind)  # 重新运行find_recs方法

    def __str__(self):
        return f"{self.type}({self.iw}, {self.ih}, {self.px_area}, '{self.judgement}')"

    def __repr__(self):
        return f'{self}'


def get_frame_grid_recursive(rect, grids, frame_mask_group, media_type):
    rec = Rect(rect, media_type)  # 创建Rect对象
    rec.put_frame_mask(frame_mask_group)  # 对当前Rect进行分割

    if rec.pg.basic:  # 如果当前Rect对象是基本单元格
        grids.append(rec)  # 将其添加到格子列表中
    else:  # 如果当前Rect对象不是基本单元格
        rects = rec.pg.rects  # 获取当前Rect对象的子矩形列表
        for r in rects:  # 对子矩形列表中的每个矩形
            get_frame_grid_recursive(r, grids, frame_mask_group, media_type)  # 递归进行处理


# @logger.catch
def get_grids(frame_mask_group, media_type):
    frame_mask = frame_mask_group[0]
    ih, iw = frame_mask.shape[0:2]  # 获取图像的高度和宽度
    rec0 = (0, 0, iw, ih)  # 定义初始矩形的坐标和大小
    pg = PicGrid(frame_mask_group, media_type)  # 创建PicGrid对象
    grids = []  # 初始化格子列表
    if pg.basic:  # 如果PicGrid对象是基本单元格
        # ================如果主图不可分割================
        rec0 = Rect(rec0, media_type)  # 创建Rect对象
        rec0.put_frame_mask(frame_mask_group)  # 对Rect对象进行分割
        grids.append(rec0)  # 将Rect对象添加到格子列表中
    else:
        # ================如果主图可以分割================
        rects = pg.rects_dic[pg.judgement]  # 获取PicGrid对象的子矩形列表
        for rect in rects:  # 对子矩形列表中的每个矩形
            get_frame_grid_recursive(rect, grids, frame_mask_group, media_type)  # 递归进行处理
    return grids  # 返回格子列表


class Line2Prev(QGraphicsLineItem):
    def __init__(self, parent: QGraphicsItem = None):
        super().__init__(parent)
        self.type = self.__class__.__name__
        self.setPen(QPen(QColor(128, 0, 128, 128), 5))

    def update_line(self, end_poly):
        self.setLine(self.parentItem().cnt.cx, self.parentItem().cnt.cy, end_poly.cnt.cx, end_poly.cnt.cy)

    def __str__(self):
        return f"{self.type}"

    def __repr__(self):
        return f'{self}'


class LabelOrd(QGraphicsItem):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.type = self.__class__.__name__
        self.text = text
        self.font = QFont('Arial', 50, QFont.Weight.Normal)
        self.textColor = QColor('#696969')
        self.outlineColor = QColor(255, 255, 255)

    def boundingRect(self):
        fm = QFontMetrics(self.font)
        return QRectF(fm.boundingRect(self.text))

    def paint(self, painter, option, widget):
        path = QPainterPath()
        path.addText(0, 0, self.font, self.text)
        painter.setBrush(self.textColor)
        painter.setPen(QPen(self.outlineColor, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)


class ContourOrd(QGraphicsTextItem):
    def __init__(self, cnt, idx, order_window, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.type = self.__class__.__name__
        self.cnt = cnt
        self.poly_idx = idx
        self.parent_window = order_window
        self.swap_color = QColor(*[int(255 * c) for c in colormap_tab20(idx % 20)[:3]])
        self.swap_color.setAlpha(128)
        self.darker_color = self.swap_color.darker(300)
        self.darker_color.setAlpha(200)
        self.setPlainText(str(idx))
        self.setDefaultTextColor(self.darker_color)
        # 设置字体
        self.setFont(QFont('Arial', 100, QFont.Weight.Normal))
        # 设置位置
        text_rect = self.boundingRect()
        self.setPos(cnt.cx - text_rect.width() / 2, cnt.cy - text_rect.height() / 2)
        self.setGraphicsEffect(
            QGraphicsDropShadowEffect(blurRadius=5, xOffset=0, yOffset=0, color=QColor(255, 255, 255, 128)))

    def update_ord(self):
        idx = self.parentItem().mode_indices[self.parent_window.order_mode]
        if idx is not None:
            self.setPlainText(str(idx))
        else:
            self.setPlainText('')


class ContourPoly(QGraphicsPolygonItem):

    def __init__(self, cnt, idx, order_window, *args, **kwargs):
        super().__init__(QPolygonF([QPointF(pt[0], pt[1]) for pt in cnt.contour.squeeze()]), *args, **kwargs)
        self.type = self.__class__.__name__
        self.cnt = cnt
        self.poly_idx = idx  # 不变
        self.idx_label = idx2label(self.poly_idx)  # 不变
        self.parent_window = order_window
        self.mode_indices = {'swap': idx, 'manual': None}  # 可变
        self.swap_indice = self.mode_indices['swap']
        self.manual_indice = self.mode_indices['manual']
        self.sel_states = {'swap': False, 'manual': False}
        self.swap_color = QColor(*[int(255 * c) for c in colormap_tab20(idx % 20)[:3]])
        self.swap_color.setAlpha(128)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPen(QPen(self.swap_color, 5))
        self.setBrush(QBrush(self.swap_color, Qt.BrushStyle.SolidPattern))

        self.centroid_dot = QGraphicsEllipseItem(self.cnt.cx - 5, self.cnt.cy - 5, 10, 10, parent=self)
        self.centroid_dot.setBrush(QBrush(QColor(0, 0, 255, 128)))
        self.line2prev = Line2Prev(parent=self)
        self.label_ord = LabelOrd(self.idx_label, parent=self)
        self.contour_ord = ContourOrd(cnt, idx, order_window, parent=self)

    def setup_default_ord(self):
        bounding_rect = self.boundingRect()
        scene_rect = self.scene().sceneRect()
        ord_width = self.label_ord.boundingRect().width()
        ord_height = self.label_ord.boundingRect().height()
        ay = bounding_rect.top() + ord_height
        if bounding_rect.left() - ord_width >= scene_rect.left():
            ax = bounding_rect.left() - ord_width
        else:
            ax = bounding_rect.right()
        self.label_ord.setPos(ax, ay)

    def update_poly(self):
        # ================序数================
        self.mode_indices['swap'] = self.parent_window.custom_cnts.index(self.cnt)
        if self.cnt in self.parent_window.manual_cnts:
            self.mode_indices['manual'] = self.parent_window.manual_cnts.index(self.cnt)
        else:
            self.mode_indices['manual'] = None
        self.swap_indice = self.mode_indices['swap']
        self.manual_indice = self.mode_indices['manual']
        self.contour_ord.update_ord()
        # ================选中与否================
        if self.sel_states[self.parent_window.order_mode]:
            self.setPen(QPen(Qt.red, 3))
        else:
            self.setPen(QPen(self.swap_color, 5))
        # ================连线================
        if self.parent_window.order_mode == 'swap':
            if self.swap_indice == 0:
                self.line2prev.update_line(self)
            else:
                prev_polygon = next((x for x in self.parent_window.contour_polys
                                     if x.swap_indice == self.swap_indice - 1), None)
                if prev_polygon:
                    self.line2prev.update_line(prev_polygon)
        elif self.parent_window.order_mode == 'manual':
            self.line2prev.update_line(self)

    def toggle_selected(self):
        current_mode = self.parent_window.order_mode
        self.sel_states[current_mode] = not self.sel_states[current_mode]
        self.update_poly()

    def mouseDoubleClickEvent(self, event):
        if self.parent_window.order_mode == 'manual' and self.cnt in self.parent_window.manual_cnts:
            self.parent_window.manual_cnts.remove(self.cnt)
            self.parent_window.update_contourpolys()
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if self.parent_window.order_mode == 'swap':
            imes = f"点击了轮廓{self.poly_idx}{self.idx_label}"
            logger.debug(imes)
            self.parent_window.status_bar.showMessage(imes)
            self.toggle_selected()
            self.sel_polygons = [x for x in self.parent_window.contour_polys if x.sel_states['swap']]
            self.sel_poly_idxs = [x.poly_idx for x in self.sel_polygons]
            # logger.debug(f"所有选中轮廓: {self.sel_poly_idxs}")
            if self.sel_states['swap'] and len(self.sel_poly_idxs) == 2:
                logger.debug(f"尝试交换{self.sel_poly_idxs[0]}和{self.sel_poly_idxs[1]}")
                self.parent_window.swap_contours(self.sel_poly_idxs[0], self.sel_poly_idxs[1])
                for x in self.sel_polygons:
                    x.toggle_selected()
        elif self.parent_window.order_mode == 'manual':
            if self.cnt not in self.parent_window.manual_cnts:
                self.parent_window.manual_cnts.append(self.cnt)
                self.parent_window.update_contourpolys()


class SearchLine(QLineEdit):
    def __init__(self, parent=None):
        super(SearchLine, self).__init__(parent)

        self.type = self.__class__.__name__
        self.parent_window = parent
        # 区分大小写按钮
        self.case_sensitive_button = ibut(self.tr('Case Sensitive'), 'msc.case-sensitive')
        # 全词匹配按钮
        self.whole_word_button = ibut(self.tr('Whole Word'), 'msc.whole-word')
        # 正则表达式按钮
        self.regex_button = ibut(self.tr('Use Regex'), 'mdi.regex')

        # 创建水平布局，用于将三个按钮放在一行
        self.hb_search_bar = QHBoxLayout()
        self.hb_search_bar.setContentsMargins(0, 0, 0, 0)
        self.hb_search_bar.setSpacing(2)
        self.hb_search_bar.addStretch()
        self.hb_search_bar.addWidget(self.case_sensitive_button)
        self.hb_search_bar.addWidget(self.whole_word_button)
        self.hb_search_bar.addWidget(self.regex_button)

        self.case_sensitive_button.clicked.connect(lambda: self.parent_window.filter_imgs(self.text()))
        self.whole_word_button.clicked.connect(lambda: self.parent_window.filter_imgs(self.text()))
        self.regex_button.clicked.connect(lambda: self.parent_window.filter_imgs(self.text()))
        self.textChanged.connect(self.parent_window.filter_imgs)

        # 设置占位符文本
        self.setPlaceholderText(self.tr('Search'))
        # 将按钮添加到 QLineEdit 的右侧
        self.setLayout(self.hb_search_bar)


class CustImageList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.type = self.__class__.__name__
        self.parent_window = parent
        self.display_mode = 0
        self.font = QFont()
        self.font.setWordSpacing(0)
        # 设置图标模式
        self.setViewMode(QListView.IconMode)
        self.setIconSize(QSize(thumb_size, thumb_size))  # 设置图标大小
        self.setResizeMode(QListView.ResizeMode.Adjust)  # 设置自动调整大小
        self.setSpacing(5)  # 设置间距
        self.setWordWrap(True)  # 开启单词换行
        self.setWrapping(False)  # 关闭自动换行
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 设置单选模式
        self.setFlow(QListView.Flow.TopToBottom)  # 设置从上到下排列
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)  # 设置自定义右键菜单
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.itemSelectionChanged.connect(self.on_img_selected)
        self.load_img_list()

    def load_img_list(self):
        # 加载图片列表
        self.clear()
        for i in range(len(self.parent_window.img_list)):
            img = self.parent_window.img_list[i]
            pixmap = QPixmap(str(img)).scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(pixmap), img.name)
            item.setFont(self.font)
            item.setData(Qt.ItemDataRole.UserRole, img)
            item.setTextAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            self.addItem(item)

    def on_img_selected(self):
        # 通过键鼠点击获取的数据是真实序数img_ind
        selected_items = self.selectedItems()
        if selected_items:
            current_item = selected_items[0]
            img_ind = self.row(current_item)
            if img_ind != self.parent_window.img_ind:
                img_file = current_item.data(Qt.ItemDataRole.UserRole)
                self.parent_window.open_img_by_path(img_file)

    def set_display_mode(self, display_mode):
        self.display_mode = display_mode
        self.setUpdatesEnabled(False)
        for index in range(self.count()):
            item = self.item(index)
            data = item.data(Qt.ItemDataRole.UserRole)
            if self.display_mode == 0:  # 仅显示缩略图
                item.setIcon(QIcon(data.as_posix()))
                item.setText('')
                self.setWordWrap(False)
            elif self.display_mode == 1:  # 仅显示文件名
                item.setIcon(QIcon())  # 清除图标
                item.setText(data.name)
                self.setWordWrap(False)  # 确保文件名不换行
            elif self.display_mode == 2:  # 同时显示缩略图和文件名
                item.setIcon(QIcon(data.as_posix()))
                item.setText(data.name)
                self.setWordWrap(True)

        self.setIconSize(QSize(thumb_size, thumb_size))  # 设置图标大小
        if self.display_mode == 1:  # 仅显示文件名
            self.setGridSize(QSize(-1, -1))  # 使用默认大小的网格
        else:
            # 为缩略图和两种模式设置网格大小
            # 为文件名和间距增加额外的宽度
            self.setGridSize(QSize(thumb_size + 30, -1))

        self.setUpdatesEnabled(True)

    def show_context_menu(self, point):
        item = self.itemAt(point)
        if item:
            context_menu = QMenu(self)
            # 创建并添加菜单项
            open_file_action = QAction(self.tr('Open in Explorer'), self)
            open_img_action = QAction(self.tr('Open in Preview'), self)
            open_with_ps = QAction(self.tr('Photoshop'), self)
            # 添加拷贝图片路径、拷贝图片名的选项
            copy_img_path = QAction(self.tr('Copy Image Path'), self)
            copy_img_name = QAction(self.tr('Copy Image Name'), self)

            context_menu.addAction(open_file_action)
            context_menu.addAction(open_img_action)
            # 添加一个打开方式子菜单
            open_with_menu = context_menu.addMenu(self.tr('Open with'))
            open_with_menu.addAction(open_with_ps)
            context_menu.addAction(copy_img_path)
            context_menu.addAction(copy_img_name)

            open_file_action.triggered.connect(lambda: open_in_explorer(item.data(Qt.ItemDataRole.UserRole)))
            open_img_action.triggered.connect(lambda: open_in_viewer(item.data(Qt.ItemDataRole.UserRole)))
            open_with_ps.triggered.connect(lambda: open_in_ps(item.data(Qt.ItemDataRole.UserRole)))
            copy_img_path.triggered.connect(lambda: copy2clipboard(item.data(Qt.ItemDataRole.UserRole).as_posix()))
            copy_img_name.triggered.connect(lambda: copy2clipboard(item.text()))

            context_menu.exec(self.mapToGlobal(point))


class CustGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.type = self.__class__.__name__
        self.img_file = None

    def load_qimg(self, image_data, img_file=None):
        self.img_file = img_file
        # 如果输入是Pillow图像，将其转换为NumPy数组
        if isinstance(image_data, Image.Image):
            image_data = array(image_data)

        # 确保输入数据是NumPy数组
        if isinstance(image_data, ndarray):
            height, width, channel = image_data.shape
            bytes_per_line = channel * width

            if channel == 4:
                # 如果输入图像有4个通道（带有Alpha通道）
                qimage_format = QImage.Format_ARGB32
            elif channel == 3:
                # 如果输入图像有3个通道
                # 如果输入图像使用BGR顺序，交换颜色通道以获得正确的RGB顺序
                image_data = cvtColor(image_data, COLOR_BGR2RGB)
                qimage_format = QImage.Format_RGB888
            else:
                # 其他情况暂不支持
                raise ValueError("Unsupported number of channels in the input image.")

            # 将NumPy数组转换为QImage
            qimage = QImage(image_data.data, width, height, bytes_per_line, qimage_format)
            # 将QImage转换为QPixmap
            pixmap = QPixmap.fromImage(qimage)
            # ================清除之前的图像================
            self.clear()
            # ================显示新图片================
            self.addPixmap(pixmap)
            # 将视图大小设置为 pixmap 的大小，并将图像放入视图中
            self.setSceneRect(pixmap.rect().toRectF())


class CustGraphicsView(QGraphicsView):
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.type = self.__class__.__name__
        self.parent_window = parent
        # 默认缩放级别，表示原始大小
        self.zoom_level = scaling_factor_reci
        self.start_pt = None
        self.selected_contours = []
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setInteractive(True)
        self.setMouseTracking(True)
        # 设置渲染、优化和视口更新模式
        # 设置渲染抗锯齿
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 设置优化标志以便不为抗锯齿调整
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # 设置优化标志以便不保存画家状态
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        # 设置视口更新模式为完整视口更新
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # 设置渲染提示为平滑图像变换，以提高图像的显示质量
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(Qt.GlobalColor.lightGray))

    def cust_zoom(self, factor):
        self.scale(factor, factor)
        self.zoom_level *= factor
        # 在主窗口中更新缩放输入
        self.parent_window.update_zoom_label()
        self.zoomChanged.emit(self.zoom_level)

    def cust_zoom_in(self):
        self.cust_zoom(1.25)

    def cust_zoom_out(self):
        self.cust_zoom(0.8)

    def fit2view(self, mode):
        rect = self.scene().itemsBoundingRect()
        view_rect = self.viewport().rect()
        scale_factor_x = view_rect.width() / rect.width()
        scale_factor_y = view_rect.height() / rect.height()
        if mode == 'width':
            scale_factor = scale_factor_x
        elif mode == 'height':
            scale_factor = scale_factor_y
        elif mode == 'screen':
            scale_factor = min(scale_factor_x, scale_factor_y)
        elif mode == 'original':
            scale_factor = scaling_factor_reci
        self.zoom_level = scale_factor
        self.resetTransform()
        self.scale(scale_factor, scale_factor)
        self.zoomChanged.emit(self.zoom_level)

    def mousePressEvent(self, event):
        self.start_pt = event.pos()
        scene_pt = self.mapToScene(self.start_pt)
        if self.scene().sceneRect().contains(scene_pt) and self.parent_window.order_mode == 'swap':
            if event.modifiers() == Qt.ShiftModifier:
                self.setDragMode(QGraphicsView.RubberBandDrag)
            else:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.parent_window.order_mode == 'swap' and self.dragMode() == QGraphicsView.RubberBandDrag:
            selected_items = self.scene().selectedItems()
            for item in selected_items:
                if isinstance(item, ContourPoly):
                    self.selected_contours.append(item)
                    item.toggle_selected()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mouseReleaseEvent(event)


class OrderWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.type = self.__class__.__name__
        self.a0_para()
        self.a1_initialize()
        self.a2_status_bar()
        self.a3_docks()
        self.a4_actions()
        self.a5_menubar()
        self.a6_toolbar()
        self.a9_setting()

    def b1_window(self):
        return

    def a0_para(self):
        # ================初始化变量================
        self.screen_icon = qicon('ei.screen')
        self.setWindowIcon(self.screen_icon)
        self.cgs = CustGraphicsScene(self)
        self.cgv = CustGraphicsView(self)
        self.cgv.setScene(self.cgs)
        self.cgv.zoomChanged.connect(self.update_zoom_label)
        self.resize(window_w, window_h)

    def a1_initialize(self):
        # ================图片列表================
        self.img_folder = img_folder
        self.auto_subdir = Auto / self.img_folder.name
        make_dir(self.auto_subdir)
        self.frame_yml = self.img_folder.parent / f'{self.img_folder.name}.yml'
        self.order_yml = self.img_folder.parent / f'{self.img_folder.name}-气泡排序.yml'
        self.cnts_dic_pkl = self.img_folder.parent / f'{self.img_folder.name}.pkl'
        self.img_list = get_valid_imgs(self.img_folder)
        self.all_masks = get_valid_imgs(self.img_folder, mode='mask')
        self.frame_data = iload_data(self.frame_yml)
        self.order_data = iload_data(self.order_yml)
        self.cnts_dic = iload_data(self.cnts_dic_pkl, mode='pkl')
        self.filter_img_list = self.img_list
        self.img_ind = clamp(img_ind, 0, len(self.img_list) - 1)
        self.img_file = self.img_list[self.img_ind]
        self.contour_polys = []
        self.setWindowTitle(self.img_file.name)

    def a2_status_bar(self):
        # ================状态栏================
        self.status_bar = QStatusBar()
        # 设置状态栏，类似布局设置
        self.setStatusBar(self.status_bar)

    def a3_docks(self):
        self.nav_tab = QTabWidget(self)
        self.cil = CustImageList(self)
        self.nav_tab.addTab(self.cil, self.tr('Thumbnails'))

        self.search_line = SearchLine(self)
        self.vb_search_nav = QVBoxLayout(self)
        self.vb_search_nav.addWidget(self.search_line)
        self.vb_search_nav.addWidget(self.nav_tab)
        self.pics_widget = QWidget()
        self.pics_widget.setLayout(self.vb_search_nav)

        self.pics_dock = QDockWidget(self.tr('Image List'), self)
        self.pics_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.pics_dock.setWidget(self.pics_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pics_dock)

    def a4_actions(self):
        self.le_scale_percent = QLineEdit(self)
        self.le_scale_percent.setFixedWidth(50)
        self.le_scale_percent.setValidator(QDoubleValidator(1, 1000, 2))

        self.open_folder_action = iact(self.tr('Open Folder'), 'ei.folder', QKeySequence.StandardKey.Open,
                                       trig=self.open_folder_by_dialog)
        self.zoom_in_action = iact(self.tr('Zoom In'), 'ei.zoom-in', QKeySequence.StandardKey.ZoomIn,
                                   trig=self.cgv.cust_zoom_in)
        self.zoom_out_action = iact(self.tr('Zoom Out'), 'ei.zoom-out', QKeySequence.StandardKey.ZoomOut,
                                    trig=self.cgv.cust_zoom_out)
        self.fit2screen_action = iact(self.tr('Fit to Screen'), 'mdi6.fit-to-screen-outline', "Ctrl+F",
                                      trig=lambda: self.cgv.fit2view("screen"))
        self.fit2width_action = iact(self.tr('Fit to Width'), 'ei.resize-horizontal', "Ctrl+W",
                                     trig=lambda: self.cgv.fit2view("width"))
        self.fit2height_action = iact(self.tr('Fit to Height'), 'ei.resize-vertical', "Ctrl+H",
                                      trig=lambda: self.cgv.fit2view("height"))
        self.reset_zoom_action = iact(self.tr('Reset Zoom'), 'mdi6.backup-restore', "Ctrl+0",
                                      trig=lambda: self.cgv.fit2view("original"))
        self.prev_img_action = iact(self.tr('Previous Image'), 'ei.arrow-left', "Ctrl+Left",
                                      trig=lambda: self.nav_img(-1))
        self.next_img_action = iact(self.tr('Next Image'), 'ei.arrow-right', "Ctrl+Right",
                                      trig=lambda: self.nav_img(1))
        self.first_img_action = iact(self.tr('First Image'), 'ei.step-backward', "Ctrl+Home",
                                       trig=lambda: self.nav_img("first"))
        self.last_img_action = iact(self.tr('Last Image'), 'ei.step-forward', "Ctrl+End",
                                      trig=lambda: self.nav_img("last"))
        self.swap_order_action = iact(self.tr('Swap Ordering'), 'mdi.swap-horizontal', checkable=True,
                                      trig=lambda: self.set_order_mode('swap'))
        self.manual_order_action = iact(self.tr('Manual Ordering'), 'fa.hand-grab-o', checkable=True,
                                        trig=lambda: self.set_order_mode('manual'))
        self.undo_action = iact(self.tr('Undo'), 'fa5s.undo', QKeySequence.StandardKey.Undo, trig=self.undo)
        self.redo_action = iact(self.tr('Redo'), 'fa5s.redo', QKeySequence.StandardKey.Redo, trig=self.redo)
        self.save_action = iact(self.tr('Save'), 'msc.save', QKeySequence.StandardKey.Save, trig=self.save2yaml)

        self.ag_sorting = QActionGroup(self)
        self.ag_sorting.addAction(self.swap_order_action)
        self.ag_sorting.addAction(self.manual_order_action)
        self.swap_order_action.setChecked(True)  # 默认选中
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.le_scale_percent.editingFinished.connect(self.scale_by_percent)
        self.update_zoom_label()

    def a5_menubar(self):
        # 文件菜单
        self.file_menu = self.menuBar().addMenu(self.tr('File'))
        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.save_action)

        # 显示菜单
        self.view_menu = self.menuBar().addMenu(self.tr('View'))
        # 视图菜单选项
        self.view_menu.addAction(self.pics_dock.toggleViewAction())
        self.view_menu.addSeparator()
        # 缩放选项
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addAction(self.fit2screen_action)
        self.view_menu.addAction(self.fit2width_action)
        self.view_menu.addAction(self.fit2height_action)
        self.view_menu.addAction(self.reset_zoom_action)
        self.view_menu.addSeparator()

        # 显示选项
        self.display_modes = [(self.tr('Show Thumbnails'), 0),
                              (self.tr('Show Filenames'), 1),
                              (self.tr('Show Both'), 2)]
        self.display_mode_group = QActionGroup(self)
        for display_mode in self.display_modes:
            action = QAction(display_mode[0], self, checkable=True)
            action.triggered.connect(lambda _, mode=display_mode[1]: self.cil.set_display_mode(mode))
            self.view_menu.addAction(action)
            self.display_mode_group.addAction(action)

        # 默认选中 Show Both 选项
        self.display_mode_group.actions()[2].setChecked(True)

        # 编辑菜单
        self.edit_menu = self.menuBar().addMenu(self.tr('Edit'))
        self.edit_menu.addAction(self.swap_order_action)
        self.edit_menu.addAction(self.manual_order_action)
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)

        # 导航菜单
        self.nav_menu = self.menuBar().addMenu(self.tr('Navigate'))
        self.nav_menu.addAction(self.prev_img_action)
        self.nav_menu.addAction(self.next_img_action)
        self.nav_menu.addAction(self.first_img_action)
        self.nav_menu.addAction(self.last_img_action)

    def a6_toolbar(self):
        self.tool_bar = QToolBar(self)
        self.tool_bar.setObjectName("Toolbar")
        self.tool_bar.setIconSize(QSize(24, 24))
        self.tool_bar.setMovable(False)
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(self.tool_bar)
        self.tool_bar.addAction(self.open_folder_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.zoom_in_action)
        self.tool_bar.addAction(self.zoom_out_action)
        self.tool_bar.addAction(self.fit2screen_action)
        self.tool_bar.addAction(self.fit2width_action)
        self.tool_bar.addAction(self.fit2height_action)
        self.tool_bar.addAction(self.reset_zoom_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.first_img_action)
        self.tool_bar.addAction(self.prev_img_action)
        self.tool_bar.addAction(self.next_img_action)
        self.tool_bar.addAction(self.last_img_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.swap_order_action)
        self.tool_bar.addAction(self.manual_order_action)
        self.tool_bar.addAction(self.undo_action)
        self.tool_bar.addAction(self.redo_action)
        self.tool_bar.addAction(self.save_action)
        self.tool_bar.addWidget(self.le_scale_percent)
        self.tool_bar.addWidget(QLabel('%'))

    def a9_setting(self):
        self.open_img_by_path(self.img_file)
        self.setCentralWidget(self.cgv)
        self.show()

    def open_img_by_path(self, img_file):
        img_file = Path(img_file)
        if img_file.exists() and img_file != self.cgs.img_file:
            self.img_file = img_file
            self.img_file_size = getsize(self.img_file)
            self.img_ind = self.img_list.index(self.img_file)
            self.setWindowTitle(self.img_file.name)
            # ================显示图片================
            self.image_raw = imdecode(fromfile(self.img_file, dtype=uint8), -1)
            self.ih, self.iw = self.image_raw.shape[0:2]
            self.cgs.load_qimg(self.image_raw, self.img_file)
            self.scale_by_percent()
            self.update_zoom_label()
            # ================将当前图片项设为选中状态================
            self.cil.blockSignals(True)  # 阻止信号
            self.cil.setCurrentRow(self.img_ind)
            self.cil.blockSignals(False)  # 恢复信号
            # ================更新状态栏信息================
            index_str = f'{self.img_ind + 1}/{len(self.img_list)}'
            meta_str = f'{self.tr("Width")}: {self.iw} {self.tr("Height")}: {self.ih} | {self.tr("File Size")}: {self.img_file_size} bytes'
            status_text = f'{index_str} | {self.tr("Filnename")}: {self.img_file.name} | {meta_str}'
            self.status_bar.showMessage(status_text)
            QApplication.processEvents()
            self.open_contour_polys()
        QApplication.processEvents()

    def open_contour_polys(self):
        self.contour_polys.clear()
        self.manual_cnts = []
        self.undo_stack = []
        self.redo_stack = []
        self.order_mode = 'swap'
        self.swap_order_action.setChecked(True)
        if self.img_file in self.cnts_dic:
            self.ordered_cnts = self.cnts_dic[self.img_file]
        else:
            img_file, self.ordered_cnts = order1pic(self.img_file, self.frame_data, self.order_data, self.all_masks,
                                                    media_type)
        self.custom_cnts = deepcopy(self.ordered_cnts)
        if self.custom_cnts:
            for idx, cnt in enumerate(self.custom_cnts):
                self.contour_poly = ContourPoly(cnt, idx, self)
                # 加入到场景中
                self.cgs.addItem(self.contour_poly)
                self.contour_poly.setup_default_ord()
                self.contour_polys.append(self.contour_poly)
                # 更新显示效果
                self.contour_poly.update_poly()

    def open_folder_by_path(self, folder_path):
        # 判断文件夹路径是否存在
        folder_path = Path(folder_path)
        if folder_path.exists():
            # 获取所有图片文件的路径
            img_list = get_valid_imgs(folder_path)
            if img_list and folder_path != self.img_folder:
                self.img_folder = folder_path
                self.auto_subdir = Auto / self.img_folder.name
                make_dir(self.auto_subdir)
                self.frame_yml = self.img_folder.parent / f'{self.img_folder.name}.yml'
                self.order_yml = self.img_folder.parent / f'{self.img_folder.name}-气泡排序.yml'
                self.cnts_dic_pkl = self.img_folder.parent / f'{self.img_folder.name}.pkl'
                self.img_list = img_list
                self.all_masks = get_valid_imgs(self.img_folder, mode='mask')
                self.frame_data = iload_data(self.frame_yml)
                self.order_data = iload_data(self.order_yml)
                self.cnts_dic = iload_data(self.cnts_dic_pkl, mode='pkl')
                self.filter_img_list = self.img_list
                self.img_ind = 0
                self.img_file = self.img_list[self.img_ind]
                # ================更新导航栏中的图片列表================
                self.cil.load_img_list()
                self.open_img_by_path(self.img_file)

    def open_folder_by_dialog(self):
        # 如果self.image_folder已经设置，使用其上一级目录作为起始目录，否则使用当前目录
        self.img_folder = Path(self.img_folder) if self.img_folder else None
        start_directory = self.img_folder.parent.as_posix() if self.img_folder else "."
        img_folder = QFileDialog.getExistingDirectory(self, self.tr('Open Folder'), start_directory)
        if img_folder:
            self.open_folder_by_path(img_folder)

    def nav_img(self, nav_step):
        cur_img_path = self.img_list[self.img_ind]
        # 检查当前图片路径是否在过滤后的图片列表中
        if cur_img_path not in self.filter_img_list:
            return
        cur_filter_ind = self.filter_img_list.index(cur_img_path)
        if nav_step == "first":
            new_filter_ind = 0
        elif nav_step == "last":
            new_filter_ind = len(self.filter_img_list) - 1
        else:
            new_filter_ind = cur_filter_ind + nav_step
        if 0 <= new_filter_ind < len(self.filter_img_list):
            new_img_file = self.filter_img_list[new_filter_ind]
            self.open_img_by_path(new_img_file)

    def filter_imgs(self, search_text: str):
        # 获取搜索框的三个条件：是否区分大小写、是否全词匹配、是否使用正则表达式
        case_sensitive = self.search_line.case_sensitive_button.isChecked()
        whole_word = self.search_line.whole_word_button.isChecked()
        use_regex = self.search_line.regex_button.isChecked()

        # 获取对应的正则表达式模式对象
        regex = get_search_regex(search_text, case_sensitive, whole_word, use_regex)
        if not regex:
            return

        # 根据正则表达式筛选图片列表
        self.filter_img_list = [img_file for img_file in self.img_list if regex.search(img_file.name)]

        # 更新缩略图列表：如果图片名匹配正则表达式，显示该项，否则隐藏
        for index in range(self.cil.count()):
            item = self.cil.item(index)
            item_text = item.text()
            item.setHidden(not bool(regex.search(item_text)))

    def update_contourpolys(self):
        for c in range(len(self.contour_polys)):
            self.contour_poly = self.contour_polys[c]
            self.contour_poly.update_poly()

    def update_zoom_label(self):
        self.le_scale_percent.setText(f'{self.cgv.zoom_level * 100:.2f}')

    def scale_by_percent(self):
        target_scale = float(self.le_scale_percent.text()) / 100
        current_scale = self.cgv.transform().m11()
        scale_factor = target_scale / current_scale
        self.cgv.scale(scale_factor, scale_factor)

    def set_order_mode(self, mode):
        self.order_mode = mode
        self.update_contourpolys()

    def swap_contours(self, poly_idx1, poly_idx2):
        poly1 = [x for x in self.contour_polys if x.poly_idx == poly_idx1][0]
        poly2 = [x for x in self.contour_polys if x.poly_idx == poly_idx2][0]

        # 保存当前的轮廓顺序，以便于后续的撤销操作
        self.undo_stack.append(self.custom_cnts.copy())
        self.redo_stack.clear()
        # 交换
        idx1 = self.custom_cnts.index(poly1.cnt)
        idx2 = self.custom_cnts.index(poly2.cnt)
        self.custom_cnts[idx1], self.custom_cnts[idx2] = self.custom_cnts[idx2], self.custom_cnts[idx1]

        # 在状态栏显示交换信息
        imes = f'交换了轮廓{poly1.idx_label}和轮廓{poly2.idx_label}'
        # logger.debug(imes)
        self.status_bar.showMessage(imes)
        # 更新撤销和重做按钮的状态
        self.undo_action.setEnabled(True)
        self.redo_action.setEnabled(False)
        self.update_contourpolys()

    def undo(self):
        if not self.undo_stack:
            return

        self.redo_stack.append(self.custom_cnts.copy())
        self.custom_cnts = self.undo_stack.pop()
        self.update_contourpolys()

        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(True)

    def redo(self):
        if not self.redo_stack:
            return

        self.undo_stack.append(self.custom_cnts.copy())
        self.custom_cnts = self.redo_stack.pop()
        self.update_contourpolys()

        self.redo_action.setEnabled(len(self.redo_stack) > 0)
        self.undo_action.setEnabled(True)

    def keyPressEvent(self, event):
        if self.order_mode == 'swap':
            if event.key() == Qt.Key_Escape:
                self.sel_polygons = [x for x in self.contour_polys if x.sel_states['swap']]
                for s in range(len(self.sel_polygons)):
                    self.contour_poly = self.sel_polygons[s]
                    self.contour_poly.toggle_selected()
                # 更新轮廓的高亮显示
                self.update_contourpolys()
                self.status_bar.showMessage('已取消所有选中的轮廓')
            else:
                self.sel_polygons = [x for x in self.contour_polys if x.sel_states['swap']]
                if self.sel_polygons:
                    direction = None
                    if event.key() == Qt.Key_Minus:
                        direction = -1
                    elif event.key() == Qt.Key_Equal:
                        direction = 1
                    if direction:
                        self.sel_polygons.sort(key=lambda x: x.swap_indice, reverse=(direction == 1))
                        boundary_check = (0 if direction == -1 else len(self.custom_cnts) - 1)
                        if self.sel_polygons[0].swap_indice != boundary_check:
                            for poly in self.sel_polygons:
                                cnt2 = self.custom_cnts[poly.swap_indice + direction]
                                poly2 = get_poly_by_cnt(self.contour_polys, cnt2)
                                self.swap_contours(poly.poly_idx, poly2.poly_idx)
                            self.update_contourpolys()
            self.sel_polygons = [x for x in self.contour_polys if x.sel_states['swap']]

    def save2yaml(self):
        if self.order_mode == 'swap':
            order2yaml(self.order_yml, self.ordered_cnts, self.custom_cnts, self.img_file)
        elif self.order_mode == 'manual':
            if len(self.manual_cnts) == len(self.custom_cnts):
                self.manual_ordered_cnts = [self.custom_cnts[x] for x in self.manual_cnts]
                order2yaml(self.order_yml, self.ordered_cnts, self.manual_ordered_cnts, self.img_file)
            else:
                self.status_bar.showMessage("在手动模式下，所有轮廓必须排序才能保存")


class MistWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.type = self.__class__.__name__
        self.a0_para()
        self.a1_initialize()
        self.a2_status_bar()
        self.a3_docks()
        self.a4_actions()
        self.a5_menubar()
        self.a6_toolbar()
        self.a9_setting()

    def b1_window(self):
        return

    def a0_para(self):
        # ================初始化变量================
        self.screen_icon = qicon('ei.screen')
        self.setWindowIcon(self.screen_icon)
        self.cgs = CustGraphicsScene(self)  # 图形场景对象
        self.cgv = CustGraphicsView(self)  # 图形视图对象
        self.cgv.setScene(self.cgs)
        self.cgv.zoomChanged.connect(self.update_zoom_label)
        self.setWindowTitle(window_title_prefix)
        self.resize(window_w, window_h)

    def a1_initialize(self):
        # ================图片列表================
        self.img_folder = None
        self.auto_subdir = None
        self.img_list = []
        self.img_file = None
        self.image = QImage()
        self.pixmap_item = QGraphicsPixmapItem()
        self.img_ind = -1
        self.display_mode = 0
        # ================最近文件夹================
        self.recent_folders = []
        # ================设置================
        # 根据需要修改组织和应用名
        self.program_settings = QSettings("MOMO", "MomoTranslator")
        self.media_type_ind = self.program_settings.value('media_type_ind', 0)
        self.media_type = media_type_dict[self.media_type_ind]
        self.media_lang_ind = self.program_settings.value('media_lang_ind', 0)
        self.media_lang = media_lang_dict[self.media_lang_ind]

    def a2_status_bar(self):
        # ================状态栏================
        self.status_bar = QStatusBar()
        # 设置状态栏，类似布局设置
        self.setStatusBar(self.status_bar)

    def a3_docks(self):
        # ================缩略图列表================
        self.nav_tab = QTabWidget(self)
        self.cil = CustImageList(self)
        self.nav_tab.addTab(self.cil, self.tr('Thumbnails'))

        self.search_line = SearchLine(self)
        self.vb_search_nav = QVBoxLayout(self)
        self.vb_search_nav.addWidget(self.search_line)
        self.vb_search_nav.addWidget(self.nav_tab)
        self.pics_widget = QWidget()
        self.pics_widget.setLayout(self.vb_search_nav)

        # 用于显示图片列表
        self.pics_dock = QDockWidget(self.tr('Image List'), self)
        self.pics_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.pics_dock.setWidget(self.pics_widget)

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
            if index == self.media_type_ind:
                self.rb_media_type.setChecked(True)

        self.hb_media_type.addStretch(1)

        # 创建语言选择下拉框
        self.cb_media_lang = QComboBox()
        self.cb_media_lang.addItems([
            self.tr("English"),
            self.tr("Chinese Simplified"),
            self.tr("Chinese Traditional"),
            self.tr("Japanese"),
            self.tr("Korean"),
        ])
        self.cb_media_lang.setCurrentIndex(self.media_lang_ind)
        self.cb_media_lang.currentIndexChanged.connect(self.update_media_lang)

        # 在布局中添加下拉框
        self.hb_media_lang = QHBoxLayout()
        self.lb_media_lang = QLabel(self.tr('Language'))
        self.lb_media_lang.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hb_media_lang.addWidget(self.lb_media_lang)
        self.hb_media_lang.addWidget(self.cb_media_lang)
        self.hb_media_lang.addStretch(1)

        self.vb_setting = QVBoxLayout()
        self.vb_setting.addWidget(self.lb_setting)
        self.vb_setting.addLayout(self.hb_media_type)
        self.vb_setting.addLayout(self.hb_media_lang)
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
        self.pb_step.clicked.connect(self.istart_task)
        self.hb_step.addWidget(self.pb_step)

        # 创建选项组
        self.task_names = [
            self.tr('Analyze Frames'),
            self.tr('Analyze Bubbles'),
            self.tr('Order Bubbles'),
            self.tr('OCR'),
            self.tr('Translate'),
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

        self.text_dock = QDockWidget(self.tr('Text'), self)
        self.text_dock.setObjectName("TextDock")
        self.text_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.text_dock.setWidget(self.text_tool)
        self.text_dock.setMinimumWidth(200)

        self.layer_dock = QDockWidget(self.tr('Layer'), self)
        self.layer_dock.setObjectName("LayerDock")
        self.layer_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.layer_dock.setWidget(self.layer_tool)
        self.layer_dock.setMinimumWidth(200)

        self.step_dock = QDockWidget(self.tr('Step'), self)
        self.step_dock.setObjectName("StepDock")
        self.step_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.step_dock.setWidget(self.step_tool)
        self.step_dock.setMinimumWidth(200)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pics_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.setting_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.text_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.step_dock)

    def a4_actions(self):
        # 添加缩放百分比输入框
        self.le_scale_percent = QLineEdit(self)
        self.le_scale_percent.setFixedWidth(50)
        self.le_scale_percent.setValidator(QDoubleValidator(1, 1000, 2))

        self.open_folder_action = iact(self.tr('Open Folder'), 'ei.folder', QKeySequence.StandardKey.Open,
                                       trig=self.open_folder_by_dialog)
        self.save_img_action = iact(self.tr('Save Image'), 'ri.image-edit-fill', QKeySequence.StandardKey.Save,
                                      trig=self.save_img)
        self.zoom_in_action = iact(self.tr('Zoom In'), 'ei.zoom-in', QKeySequence.StandardKey.ZoomIn,
                                   trig=self.cgv.cust_zoom_in)
        self.zoom_out_action = iact(self.tr('Zoom Out'), 'ei.zoom-out', QKeySequence.StandardKey.ZoomOut,
                                    trig=self.cgv.cust_zoom_out)
        self.fit2screen_action = iact(self.tr('Fit to Screen'), 'mdi6.fit-to-screen-outline', "Ctrl+F",
                                      trig=lambda: self.cgv.fit2view("screen"))
        self.fit2width_action = iact(self.tr('Fit to Width'), 'ei.resize-horizontal', "Ctrl+W",
                                     trig=lambda: self.cgv.fit2view("width"))
        self.fit2height_action = iact(self.tr('Fit to Height'), 'ei.resize-vertical', "Ctrl+H",
                                      trig=lambda: self.cgv.fit2view("height"))
        self.reset_zoom_action = iact(self.tr('Reset Zoom'), 'mdi6.backup-restore', "Ctrl+0",
                                      trig=lambda: self.cgv.fit2view("original"))
        self.prev_img_action = iact(self.tr('Previous Image'), 'ei.arrow-left', "Ctrl+Left",
                                      trig=lambda: self.nav_img(-1))
        self.next_img_action = iact(self.tr('Next Image'), 'ei.arrow-right', "Ctrl+Right",
                                      trig=lambda: self.nav_img(1))
        self.first_img_action = iact(self.tr('First Image'), 'ei.step-backward', "Ctrl+Home",
                                       trig=lambda: self.nav_img("first"))
        self.last_img_action = iact(self.tr('Last Image'), 'ei.step-forward', "Ctrl+End",
                                      trig=lambda: self.nav_img("last"))
        self.about_action = iact(f"{self.tr('About')} {APP_NAME}", None,
                                 trig=self.show_about_dialog)
        self.about_qt_action = iact(f"{self.tr('About')} Qt", None,
                                    trig=QApplication.instance().aboutQt)
        self.help_document_action = iact(f"{APP_NAME} {self.tr('Help')}", None,
                                         trig=self.show_help_document)
        self.feedback_action = iact('Bug Report', None, trig=self.show_feedback_dialog)
        self.update_action = iact('Update Online', None, trig=self.check_for_updates)

        self.le_scale_percent.editingFinished.connect(self.scale_by_percent)
        self.update_zoom_label()

    def a5_menubar(self):
        # 文件菜单
        self.file_menu = self.menuBar().addMenu(self.tr('File'))
        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.save_img_action)
        self.file_menu.addSeparator()
        # 添加最近打开文件夹菜单项
        self.recent_folders_menu = self.file_menu.addMenu(self.tr('Recent Folders'))
        self.update_recent_folders_menu()

        # 显示菜单
        self.view_menu = self.menuBar().addMenu(self.tr('View'))
        # 添加视图菜单选项
        self.view_menu.addAction(self.pics_dock.toggleViewAction())
        self.view_menu.addAction(self.setting_dock.toggleViewAction())
        self.view_menu.addAction(self.text_dock.toggleViewAction())
        self.view_menu.addAction(self.layer_dock.toggleViewAction())
        self.view_menu.addAction(self.step_dock.toggleViewAction())
        self.view_menu.addSeparator()

        # 添加缩放选项
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.fit2screen_action)
        self.view_menu.addAction(self.fit2width_action)
        self.view_menu.addAction(self.fit2height_action)
        self.view_menu.addAction(self.reset_zoom_action)
        self.view_menu.addSeparator()

        # 添加显示选项
        self.display_modes = [(self.tr('Show Thumbnails'), 0),
                              (self.tr('Show Filenames'), 1),
                              (self.tr('Show Both'), 2)]
        self.display_mode_group = QActionGroup(self)
        for display_mode in self.display_modes:
            action = QAction(display_mode[0], self, checkable=True)
            action.triggered.connect(lambda _, mode=display_mode[1]: self.cil.set_display_mode(mode))
            self.view_menu.addAction(action)
            self.display_mode_group.addAction(action)

        # 默认选中 Show Both 选项
        self.display_mode_group.actions()[2].setChecked(True)

        # 编辑菜单
        self.edit_menu = self.menuBar().addMenu(self.tr('Edit'))

        # 导航菜单
        self.nav_menu = self.menuBar().addMenu(self.tr('Navigate'))
        self.nav_menu.addAction(self.prev_img_action)
        self.nav_menu.addAction(self.next_img_action)
        self.nav_menu.addAction(self.first_img_action)
        self.nav_menu.addAction(self.last_img_action)

        # 帮助菜单
        self.help_menu = self.menuBar().addMenu(self.tr('Help'))
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.about_qt_action)
        self.help_menu.addAction(self.help_document_action)
        self.help_menu.addAction(self.feedback_action)
        self.help_menu.addAction(self.update_action)

    def a6_toolbar(self):

        # 添加顶部工具栏
        self.tool_bar = QToolBar(self.tr('Toolbar'), self)
        self.tool_bar.setObjectName("Toolbar")
        self.tool_bar.setIconSize(QSize(24, 24))
        self.tool_bar.setMovable(False)
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)
        self.tool_bar.addAction(self.open_folder_action)
        self.tool_bar.addAction(self.save_img_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.zoom_in_action)
        self.tool_bar.addAction(self.zoom_out_action)
        self.tool_bar.addAction(self.fit2screen_action)
        self.tool_bar.addAction(self.fit2width_action)
        self.tool_bar.addAction(self.fit2height_action)
        self.tool_bar.addAction(self.reset_zoom_action)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.first_img_action)
        self.tool_bar.addAction(self.prev_img_action)
        self.tool_bar.addAction(self.next_img_action)
        self.tool_bar.addAction(self.last_img_action)
        self.tool_bar.addWidget(self.le_scale_percent)
        self.tool_bar.addWidget(QLabel('%'))

    def a9_setting(self):
        # 读取上一次打开的文件夹、窗口位置和状态
        last_opened_folder = self.program_settings.value('last_opened_folder', '')
        geometry = self.program_settings.value('window_geometry')
        state = self.program_settings.value('window_state')
        # 如果上一次打开的文件夹存在，则打开它
        if last_opened_folder and exists(last_opened_folder) and isdir(last_opened_folder):
            self.open_folder_by_path(last_opened_folder)
        # 如果上一次有记录窗口位置，则恢复窗口位置
        if geometry is not None:
            self.restoreGeometry(geometry)
        # 如果上一次有记录窗口状态，则恢复窗口状态
        if state is not None:
            self.restoreState(state)

        # 将 QGraphicsView 设置为中心窗口部件
        self.setCentralWidget(self.cgv)

        # 显示窗口
        self.show()

    def open_img_by_path(self, img_file):
        img_file = Path(img_file)
        if img_file.exists() and img_file != self.cgs.img_file:
            self.img_file = img_file
            self.img_file_size = getsize(self.img_file)
            self.img_ind = self.img_list.index(self.img_file)
            # ================显示图片================
            self.image_raw = imdecode(fromfile(self.img_file, dtype=uint8), -1)
            self.ih, self.iw = self.image_raw.shape[0:2]
            self.cgs.load_qimg(self.image_raw, self.img_file)
            self.scale_by_percent()
            self.update_zoom_label()
            # ================将当前图片项设为选中状态================
            self.cil.blockSignals(True)  # 阻止信号
            self.cil.setCurrentRow(self.img_ind)
            self.cil.blockSignals(False)  # 恢复信号
            # ================更新状态栏信息================
            index_str = f'{self.img_ind + 1}/{len(self.img_list)}'
            meta_str = f'{self.tr("Width")}: {self.iw} {self.tr("Height")}: {self.ih} | {self.tr("File Size")}: {self.img_file_size} bytes'
            status_text = f'{index_str} | {self.tr("Filnename")}: {self.img_file.name} | {meta_str}'
            self.status_bar.showMessage(status_text)

    def open_folder_by_path(self, folder_path):
        # 打开上次关闭程序时用到的文件夹
        # 打开最近的文件夹
        # 打开文件夹

        # 判断文件夹路径是否存在
        folder_path = Path(folder_path)
        if folder_path.exists():
            # 获取所有图片文件的路径
            img_list = get_valid_imgs(folder_path)
            if img_list:
                self.img_folder = folder_path
                self.auto_subdir = Auto / self.img_folder.name
                self.img_list = img_list
                self.filter_img_list = self.img_list
                self.img_ind = 0
                self.img_file = self.img_list[self.img_ind]

                # ================更新导航栏中的图片列表================
                self.cil.load_img_list()
                self.open_img_by_path(self.img_file)

                self.setWindowTitle(f'{window_title_prefix} - {self.img_folder}')
                # 将该文件夹添加到最近使用文件夹列表
                self.add_recent_folder(self.img_folder)

    def open_folder_by_dialog(self):
        # 如果self.image_folder已经设置，使用其上一级目录作为起始目录，否则使用当前目录
        self.img_folder = Path(self.img_folder) if self.img_folder else None
        start_directory = self.img_folder.parent.as_posix() if self.img_folder else "."
        img_folder = QFileDialog.getExistingDirectory(self, self.tr('Open Folder'), start_directory)
        if img_folder:
            self.open_folder_by_path(img_folder)

    def update_recent_folders_menu(self):
        self.recent_folders_menu.clear()
        recent_folders = self.program_settings.value("recent_folders", [])
        for folder in recent_folders:
            action = QAction(str(folder), self)
            action.triggered.connect(lambda checked, p=folder: self.open_folder_by_path(p))
            self.recent_folders_menu.addAction(action)

    def add_recent_folder(self, folder_path):
        recent_folders = self.program_settings.value("recent_folders", [])
        recent_folders = reduce_list(recent_folders)
        if folder_path in recent_folders:
            recent_folders.remove(folder_path)
        recent_folders.insert(0, folder_path)
        # 保留最多10个最近文件夹
        recent_folders = recent_folders[:10]

        self.program_settings.setValue("recent_folders", recent_folders)
        self.update_recent_folders_menu()

    def nav_img(self, nav_step):
        cur_img_path = self.img_list[self.img_ind]
        # 检查当前图片路径是否在过滤后的图片列表中
        if cur_img_path not in self.filter_img_list:
            return
        cur_filter_ind = self.filter_img_list.index(cur_img_path)
        if nav_step == "first":
            new_filter_ind = 0
        elif nav_step == "last":
            new_filter_ind = len(self.filter_img_list) - 1
        else:
            new_filter_ind = cur_filter_ind + nav_step
        if 0 <= new_filter_ind < len(self.filter_img_list):
            new_img_file = self.filter_img_list[new_filter_ind]
            self.open_img_by_path(new_img_file)

    def filter_imgs(self, search_text: str):
        # 获取搜索框的三个条件：是否区分大小写、是否全词匹配、是否使用正则表达式
        case_sensitive = self.search_line.case_sensitive_button.isChecked()
        whole_word = self.search_line.whole_word_button.isChecked()
        use_regex = self.search_line.regex_button.isChecked()

        # 获取对应的正则表达式模式对象
        regex = get_search_regex(search_text, case_sensitive, whole_word, use_regex)
        if not regex:
            return

        # 根据正则表达式筛选图片列表
        self.filter_img_list = [img_file for img_file in self.img_list if regex.search(img_file.name)]

        # 更新缩略图列表：如果图片名匹配正则表达式，显示该项，否则隐藏
        for index in range(self.cil.count()):
            item = self.cil.item(index)
            item_text = item.text()
            item.setHidden(not bool(regex.search(item_text)))

    def save_img(self):
        file_name, _ = QFileDialog.getSaveFileName(self, self.tr('Save Image'), "", "Images (*.png *.xpm *.jpg)")
        if file_name:
            image = QImage(self.cgs.sceneRect().size().toSize(), QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            self.cgs.render(painter)
            painter.end()
            image.save(file_name)

    def update_zoom_label(self):
        self.le_scale_percent.setText(f'{self.cgv.zoom_level * 100:.2f}')

    def scale_by_percent(self):
        target_scale = float(self.le_scale_percent.text()) / 100
        current_scale = self.cgv.transform().m11()
        scale_factor = target_scale / current_scale
        self.cgv.scale(scale_factor, scale_factor)

    def show_about_dialog(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle(f"{self.tr('About')} {APP_NAME}")

        app_name_label = QLabel(APP_NAME)
        app_name_label.setFont(QFont("Arial", 20, QFont.Bold))

        version_label = QLabel(f"版本: {APP_VERSION}")
        libraries_label = QLabel("使用的库：PyQt6")
        license_label = QLabel("许可证: MIT License")

        vb_dialog = QVBoxLayout()
        vb_dialog.addWidget(app_name_label)
        vb_dialog.addWidget(version_label)
        vb_dialog.addWidget(libraries_label)
        vb_dialog.addWidget(license_label)
        about_dialog.setLayout(vb_dialog)
        about_dialog.exec_()

    def show_help_document(self):
        # 在此处显示帮助文档
        pass

    def show_feedback_dialog(self):
        recipient_email = "annebing@qq.com"
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
    # @logger.catch
    def step0_analyze_frames(self):
        frame_data = iload_data(self.frame_yml)
        # ================分析画格================
        processed_imgs = 0
        for p, img_file in enumerate(self.img_list):
            img_file, frame_grid_strs = analyze1frame(img_file, frame_data, self.auto_subdir, media_type)
            frame_data[img_file.name] = frame_grid_strs

            processed_imgs += 1
            self.pb_task.setValue(int(processed_imgs / len(self.img_list) * 100))
            QApplication.processEvents()

            pprint(frame_grid_strs)
            if len(frame_grid_strs) >= 2:
                grid_masks, marked_frames = get_grid_masks(img_file, frame_grid_strs)
                marked_frames_jpg = self.auto_subdir / f'{img_file.stem}-画格.jpg'
                marked_frames = imdecode(fromfile(marked_frames_jpg, dtype=uint8), -1)
                self.cgs.load_qimg(marked_frames, marked_frames_jpg)
                QApplication.processEvents()

        self.pb_task.setValue(100)
        QApplication.processEvents()
        frame_data_sorted = {k: frame_data[k] for k in natsorted(frame_data)}
        with open(self.frame_yml, 'w') as yml_file:
            yaml.dump(frame_data_sorted, yml_file)

    @timer_decorator
    # @logger.catch
    def step1_analyze_bubbles(self):
        frame_data = iload_data(self.frame_yml)
        processed_imgs = 0
        all_masks_old = get_valid_imgs(self.img_folder, mode='mask')

        for p, img_file in enumerate(self.img_list):
            logger.warning(f'{img_file=}')
            image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
            image_raw = toBGR(image_raw)
            ih, iw = image_raw.shape[0:2]
            # ================矩形画格信息================
            frame_grid_strs = frame_data.get(img_file.name, [f'0,0,{iw},{ih}~0,0,{iw},{ih}'])
            # ================模型检测文字，文字显示为白色================
            if use_torch and CTD_model is not None:
                CTD_mask = get_CTD_mask(image_raw)
            else:
                CTD_mask = None
            # ================针对每一张图================
            for c in range(len(color_patterns)):
                # ================遍历每种气泡文字颜色组合================
                color_pattern = color_patterns[c]
                cp_mask_cnt_png = get_bubbles_by_cp(img_file, color_pattern, frame_grid_strs, CTD_mask, media_type,
                                                    self.auto_subdir)
                if cp_mask_cnt_png.exists():
                    transparent_img = imdecode(fromfile(cp_mask_cnt_png, dtype=uint8), -1)
                    self.cgs.load_qimg(transparent_img, cp_mask_cnt_png)
                QApplication.processEvents()

            processed_imgs += 1
            self.pb_task.setValue(int(processed_imgs / len(self.img_list) * 100))
            QApplication.processEvents()

        # ================搬运气泡蒙版================
        auto_all_masks = get_valid_imgs(self.auto_subdir, mode='mask')
        # 如果步骤开始前没有气泡蒙版
        if not all_masks_old:
            for mask_src in auto_all_masks:
                mask_dst = self.img_folder / mask_src.name
                copy2(mask_src, mask_dst)

        # Update progress bar to 100%
        self.pb_task.setValue(100)
        QApplication.processEvents()

    @timer_decorator
    # @logger.catch
    def step2_order(self):
        frame_data = iload_data(self.frame_yml)
        order_data = iload_data(self.order_yml)
        all_masks = get_valid_imgs(self.img_folder, mode='mask')
        processed_imgs = 0

        for p, img_file in enumerate(self.img_list):
            logger.warning(f'{img_file=}')
            image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
            ih, iw = image_raw.shape[0:2]

            # ================矩形画格信息================
            # 从 frame_data 中获取画格信息，默认为整个图像的矩形区域
            frame_grid_strs = frame_data.get(img_file.name, [f'0,0,{iw},{ih}~0,0,{iw},{ih}'])
            bubble_order_strs = order_data.get(img_file.name, [])
            grid_masks, marked_frames = get_grid_masks(img_file, frame_grid_strs)
            order_preview_jpg = self.auto_subdir / f'{img_file.stem}-气泡排序.jpg'

            # ================获取对应的文字图片================
            mask_pics = [x for x in all_masks if x.stem.startswith(img_file.stem)]
            if mask_pics:
                single_cnts = get_single_cnts(image_raw, mask_pics)
                logger.debug(f'{len(single_cnts)=}')
                single_cnts_grids = get_ordered_cnts(single_cnts, image_raw, grid_masks, bubble_order_strs, media_type)
                ordered_cnts = list(chain(*single_cnts_grids))
                order_preview = get_order_preview(marked_frames, single_cnts_grids)
                write_pic(order_preview_jpg, order_preview)
                self.cgs.load_qimg(order_preview, order_preview_jpg)

            processed_imgs += 1
            self.pb_task.setValue(int(processed_imgs / len(self.img_list) * 100))
            QApplication.processEvents()

        # Update progress bar to 100%
        self.pb_task.setValue(100)
        QApplication.processEvents()

    def step3_OCR(self):
        frame_data = iload_data(self.frame_yml)
        order_data = iload_data(self.order_yml)
        ocr_data = iload_data(self.ocr_yml)
        # ================分析画格================
        processed_imgs = 0
        # ================气泡蒙版================
        all_masks = get_valid_imgs(self.img_folder, mode='mask')
        ocr_doc = Document()
        for i in range(len(self.img_list)):
            img_file = self.img_list[i]
            logger.warning(f'{img_file=}')
            pic_results = ocr1pic(img_file, frame_data, order_data, ocr_data, all_masks, media_type, media_lang, vert)
            ocr_doc, all_cropped_imgs = update_ocr_doc(ocr_doc, pic_results, ocr_yml, i, self.img_list)

            self.pb_task.setValue(int(processed_imgs / len(self.img_list) * 100))
            QApplication.processEvents()
            processed_imgs += 1

        # ================保存为DOCX================
        write_docx(self.ocr_docx, ocr_doc)
        self.pb_task.setValue(100)

    def istart_task(self, checked):
        # 当使用 QPushButton 的 clicked 信号连接到槽时，它会传递一个布尔值表示按钮是否被选中
        # 检查哪个 QRadioButton 被选中并执行相应的任务
        self.task_name = self.bg_step.checkedButton().text()
        self.task_ind = self.bg_step.checkedId()
        logger.debug(f'[{self.task_ind}]{self.task_name} - {self.img_folder}')
        # 在这里执行您的任务逻辑
        if self.img_folder is not None:
            self.img_folder = Path(self.img_folder)
            if self.img_folder.exists():
                self.frame_yml = self.img_folder.parent / f'{self.img_folder.name}.yml'
                self.order_yml = self.img_folder.parent / f'{self.img_folder.name}-气泡排序.yml'
                self.ocr_yml = self.img_folder.parent / f'{self.img_folder.name}-文字识别.yml'
                self.ocr_docx = self.img_folder.parent / f'{img_folder.name}-1识别.docx'
                self.auto_subdir = Auto / self.img_folder.name
                make_dir(self.auto_subdir)

                if self.task_ind == 0:
                    logger.debug(f'self.step0_analyze_frames()')
                    self.step0_analyze_frames()
                elif self.task_ind == 1:
                    self.step1_analyze_bubbles()
                elif self.task_ind == 2:
                    self.step2_order()
                elif self.task_ind == 3:
                    self.step3_OCR()
                elif self.task_ind == 4:
                    step5_translate(self.img_folder)

                # 自动选中下一个步骤，除非当前步骤是最后一个步骤
                if self.task_ind < self.bg_step.id(self.bg_step.buttons()[-1]):
                    next_task_ind = self.task_ind + 1
                    next_task_button = self.bg_step.button(next_task_ind)
                    if next_task_button:
                        next_task_button.setChecked(True)
                        self.task_ind = next_task_ind

        QApplication.processEvents()

    def update_media_type(self, button):
        index = self.bg_media_type.id(button)
        self.media_type_ind = index
        self.media_type = media_type_dict[index]

    def update_media_lang(self, index):
        self.media_lang_ind = index
        self.media_lang = media_lang_dict[index]

    def closeEvent(self, event):
        # 如果有打开图片的文件夹，将其保存到程序设置中
        if self.img_folder:
            self.program_settings.setValue('last_opened_folder', self.img_folder)
        else:
            self.program_settings.setValue('last_opened_folder', '')
        # 保存窗口的几何和状态信息到程序设置中
        self.program_settings.setValue('window_geometry', self.saveGeometry())
        self.program_settings.setValue('window_state', self.saveState())
        self.program_settings.setValue('media_type_ind', self.media_type_ind)
        self.program_settings.setValue('media_lang_ind', self.media_lang_ind)
        event.accept()


@logger.catch
def order_qt(appgui):
    order_window = OrderWindow()
    sys.exit(appgui.exec())


@logger.catch
def mist_qt(appgui):
    mist_window = MistWindow()
    sys.exit(appgui.exec())


# @logger.catch
def get_line_segments(mask, target_color='ffffff'):
    """
    根据给定的掩码获取非目标颜色的线段。
    每个线段都表示为一个 (start, end) 元组。
    """
    # 根据目标颜色设置目标值
    if target_color == 'ffffff':
        target_value = 255
    elif target_color == '000000':
        target_value = 0
    else:
        raise ValueError("不支持的颜色。")

    segments = []
    start = None
    # 遍历掩码，找到非目标颜色的线段
    for i, value in enumerate(mask):
        if value != target_value:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i))
                start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


# @logger.catch
def get_combined_mask(slice1, slice2, slice3, lower_bound, upper_bound):
    mask1 = inRange(slice1, lower_bound, upper_bound)
    mask2 = inRange(slice2, lower_bound, upper_bound)
    mask3 = inRange(slice3, lower_bound, upper_bound)
    combined_mask = bitwise_and(bitwise_and(mask1, mask2), mask3)
    return combined_mask


# @logger.catch
def get_added_frames(frame_grid_strs, image_raw, color_name0):
    directions = ['top', 'bottom', 'left', 'right']
    # 根据边框颜色获取绘制用颜色
    if color_name0 == 'white':
        pic_frame_color = 'ffffff'
        color = color_black
        lower_bound = lower_bound_white
        upper_bound = upper_bound_white
    else:
        pic_frame_color = '000000'
        color = color_white
        lower_bound = lower_bound_black
        upper_bound = upper_bound_black
    target_color = pic_frame_color
    logger.debug(f'{pic_frame_color=}')

    for frame_grid_str in frame_grid_strs:
        int_values = list(map(int, findall(r'\d+', frame_grid_str)))
        x, y, w, h, xx, yy, ww, hh = int_values

        masks = {
            'top': get_combined_mask(image_raw[yy + 2:yy + 3, xx:xx + ww],
                                     image_raw[yy + 3:yy + 4, xx:xx + ww],
                                     image_raw[yy + 4:yy + 5, xx:xx + ww],
                                     lower_bound, upper_bound),
            'bottom': get_combined_mask(image_raw[yy + hh - 3:yy + hh - 2, xx:xx + ww],
                                        image_raw[yy + hh - 2:yy + hh - 1, xx:xx + ww],
                                        image_raw[yy + hh - 1:yy + hh, xx:xx + ww],
                                        lower_bound, upper_bound),
            'left': get_combined_mask(image_raw[yy:yy + hh, xx + 1:xx + 2],
                                      image_raw[yy:yy + hh, xx + 2:xx + 3],
                                      image_raw[yy:yy + hh, xx + 3:xx + 4],
                                      lower_bound, upper_bound),
            'right': get_combined_mask(image_raw[yy:yy + hh, xx + ww - 3:xx + ww - 2],
                                       image_raw[yy:yy + hh, xx + ww - 2:xx + ww - 1],
                                       image_raw[yy:yy + hh, xx + ww - 1:xx + ww],
                                       lower_bound, upper_bound)
        }

        masks_segments = {
            direction: get_line_segments(squeeze(mask), target_color=target_color) for direction, mask in masks.items()
        }

        for direction, segments in masks_segments.items():
            total_length = ww if direction in ['top', 'bottom'] else hh
            non_white_length = sum([(seg[1] - seg[0]) for seg in segments])
            percentage = non_white_length / total_length
            gap_value = gaps[directions.index(direction)]

            if percentage <= frame_thres:
                if gap_value == 0:
                    draw_points = {
                        'top': ((xx, yy), (xx + ww, yy)),
                        'bottom': ((xx, yy + hh), (xx + ww, yy + hh)),
                        'left': ((xx, yy), (xx, yy + hh)),
                        'right': ((xx + ww, yy), (xx + ww, yy + hh))
                    }
                    line(image_raw, draw_points[direction][0], draw_points[direction][1], color, 2)
                else:
                    for offset in range(-2 if direction in ['top', 'bottom'] else -1, gap_value + 1):
                        for start, end in segments:
                            draw_points = {
                                'top': ((xx + start, yy - offset), (xx + end, yy - offset)),
                                'bottom': ((xx + start, yy + hh + offset), (xx + end, yy + hh + offset)),
                                'left': ((xx - offset, yy + start), (xx - offset, yy + end)),
                                'right': ((xx + ww + offset, yy + start), (xx + ww + offset, yy + end))
                            }
                            line(image_raw, draw_points[direction][0], draw_points[direction][1], color, 1)

                    # 绘制外侧扩大的矩形，考虑gaps延长线段
                    draw_points = {
                        'top': ((xx - gaps[2], yy - gaps[0]), (xx + ww + gaps[3], yy - gaps[0])),
                        'bottom': ((xx - gaps[2], yy + hh + gaps[1]), (xx + ww + gaps[3], yy + hh + gaps[1])),
                        'left': ((xx - gaps[2], yy - gaps[0]), (xx - gaps[2], yy + hh + gaps[1])),
                        'right': ((xx + ww + gaps[3], yy - gaps[0]), (xx + ww + gaps[3], yy + hh + gaps[1]))
                    }
                    line(image_raw, draw_points[direction][0], draw_points[direction][1], color, 2)
    return image_raw


# @logger.catch
def compute_frame_mask(image_raw, dominant_colors, tolerance):
    masks = []
    for i, dominant_color in enumerate(dominant_colors):
        dominant_color_int32 = array(dominant_color[:3]).astype(int32)
        lower_bound = maximum(0, dominant_color_int32 - tolerance)
        upper_bound = minimum(255, dominant_color_int32 + tolerance)
        mask = inRange(image_raw, lower_bound, upper_bound)
        masks.append(mask)
    # 将多个掩码相加
    combined_mask = np.sum(masks, axis=0)
    # 将 combined_mask 转换为 uint8 类型
    combined_mask_uint8 = combined_mask.astype(np.uint8)
    bubble_mask = combined_mask_uint8
    letter_mask = bitwise_not(bubble_mask)
    filter_cnts = get_raw_bubbles(bubble_mask, letter_mask, None, None, None)
    for f in range(len(filter_cnts)):
        bubble_cnt = filter_cnts[f]
        combined_mask_uint8 = drawContours(combined_mask_uint8, [bubble_cnt.contour], 0, 0, -1)
    return combined_mask_uint8


# @logger.catch
def get_proj_img(image_array, direction, target_color="black"):
    """
    根据输入的二维图像数组和投影方向，创建投影图像并返回白色像素的投影。

    :param image_array: 一个表示二值化图像的二维数组，形状为 (height, width)。
    :param direction: 投影方向，可以是 "horizontal" 或 "vertical"。
    :param target_color: 目标颜色，可以是 "black" 或 "white"。
    :return: 投影图像，以及一个列表，列表的每个元素是每行（或每列）的白色像素数量。
    """
    height, width = image_array.shape[:2]
    is_target_black = target_color.lower() == "black"

    target_value = 0 if is_target_black else 255
    image_bg_color = 255 if is_target_black else 0
    image_fg_color = 0 if is_target_black else 255

    projection_data = []
    if direction.lower() == "horizontal":
        # 计算水平投影
        projection_data = [row.count(target_value) for row in image_array.tolist()]
        # 创建水平投影图像
        proj_img = Image.new('L', (width, height), image_bg_color)
        for y, value in enumerate(projection_data):
            for x in range(value):
                proj_img.putpixel((x, y), image_fg_color)
    elif direction.lower() == "vertical":
        # 计算垂直投影
        projection_data = [col.count(target_value) for col in zip(*image_array.tolist())]
        # 创建垂直投影图像
        proj_img = Image.new('L', (width, height), image_bg_color)
        for x, value in enumerate(projection_data):
            for y in range(height - value, height):
                proj_img.putpixel((x, y), image_fg_color)
    else:
        raise ValueError("Invalid projection direction")

    return proj_img, projection_data


# @logger.catch
def analyze1frame(img_file, frame_data, auto_subdir, media_type):
    with_frames_jpg = img_file.parent / f'{img_file.stem}-画框.jpg'
    added_frames_jpg = auto_subdir / f'{img_file.stem}-加框.jpg'
    logger.warning(f'{img_file=}')
    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    image_formal = image_raw.copy()
    if with_frames_jpg.exists():
        image_formal = imdecode(fromfile(with_frames_jpg, dtype=uint8), -1)
    image_formal = toBGR(image_formal)

    color_name0 = 'white'
    if img_file.name in frame_data:
        frame_grid_strs = frame_data[img_file.name]
    else:
        if frame_color is None:
            # 获取RGBA格式的边框像素
            edge_pixels_rgba = get_edge_pxs(image_raw)
            # 只检查前两种主要颜色
            color_and_ratios, combined_color_and_ratios = find_dominant_colors(edge_pixels_rgba)
            dominant_color0, color_ratio0 = combined_color_and_ratios[0]
            color_name0 = get_color_name(dominant_color0)
            dominant_colors = [dominant_color0]
            imes = f"{img_file.stem}边框颜色中出现次数最多的颜色：{dominant_color0}, 颜色名称：{color_name0}, {color_ratio0=:.4f}"
            if check_second_color and len(combined_color_and_ratios) >= 2:
                # 第二种颜色较多且不为黑色
                dominant_color1, color_ratio1 = combined_color_and_ratios[1]
                color_name1 = get_color_name(dominant_color1)
                if color_name1 != 'black' and color_ratio1 >= 0.1:
                    dominant_colors = [dominant_color0, dominant_color1]
                    imes += f", 次多的颜色：{dominant_color1}, 颜色名称：{color_name1}, {color_ratio1=:.4f}"
        else:
            # ================从配置中读取颜色================
            dominant_color0 = color2rgb(f'#{frame_color}') + (255,)
            color_name0 = get_color_name(dominant_color0)
            dominant_colors = [dominant_color0]
            imes = f"{img_file.stem}边框颜色为：{dominant_color0}, 颜色名称：{color_name0}"
        logger.info(imes)

        # 计算主要颜色遮罩
        frame_mask = compute_frame_mask(image_formal, dominant_colors, tolerance=normal_tolerance)
        frame_mask_white = compute_frame_mask(image_formal, [rgba_white], tolerance=white_tolerance)
        frame_mask_black = compute_frame_mask(image_formal, [rgba_black], tolerance=black_tolerance)

        frame_mask_group = [frame_mask]
        if check_more_frame_color:
            if color_name0 != 'white':
                frame_mask_group.append(frame_mask_white)
            if color_name0 != 'black':
                frame_mask_group.append(frame_mask_black)

        # 创建水平投影图像
        hori_proj_frame_mask, hori_proj_data = get_proj_img(frame_mask, 'horizontal', 'white')
        # 创建垂直投影图像
        ver_proj_frame_mask, ver_proj_data = get_proj_img(frame_mask, 'vertical', 'white')

        if frame_color is not None:
            thres = 300
            # 计算一阶差分
            dif = diff(hori_proj_data)
            # 找到大于等于阈值的一阶差分的索引
            high_diff_indices = where(np.abs(dif) >= thres)[0]
            # 使用 argrelextrema 函数找到大于thres的一阶差分的所有局部最大值的索引
            high_diff_maxima = argrelextrema(dif[high_diff_indices], greater)[0]
            # 注意high_diff_maxima返回的是high_diff_indices的局部索引，我们需要把它转换成hori_proj_data的索引
            high_diff_maxima = high_diff_indices[high_diff_maxima]
            logger.info(f'{high_diff_indices=}')
            logger.info(f'{high_diff_maxima=}')

        if do_dev_pic:
            frame_mask_png = current_dir / 'frame_mask.png'
            frame_mask_white_png = current_dir / 'frame_mask_white.png'
            frame_mask_black_png = current_dir / 'frame_mask_black.png'
            hori_proj_frame_mask_png = current_dir / 'hori_proj_frame_mask.png'
            ver_proj_frame_mask_png = current_dir / 'ver_proj_frame_mask.png'
            write_pic(frame_mask_png, frame_mask)
            write_pic(frame_mask_white_png, frame_mask_white)
            write_pic(frame_mask_black_png, frame_mask_black)
            write_pic(hori_proj_frame_mask_png, hori_proj_frame_mask)
            write_pic(ver_proj_frame_mask_png, ver_proj_frame_mask)

        grids = get_grids(frame_mask_group, media_type)
        frame_grid_strs = [rect.frame_grid_str for rect in grids]
        frame_data[img_file.name] = frame_grid_strs
    pprint(frame_grid_strs)
    # 对画格进行标记
    grid_masks, marked_frames = get_grid_masks(img_file, frame_grid_strs)

    if do_add_frame:
        added_frames = get_added_frames(frame_grid_strs, image_raw, color_name0)
        write_pic(added_frames_jpg, added_frames)
    return img_file, frame_grid_strs


# @timer_decorator
def step0_analyze_frames(img_folder, frame_yml, media_type, auto_subdir, image_inds):
    """
    分析画格，获取画格的位置和尺寸，将结果保存到YAML文件中。

    :param frame_yml: frame_yml文件路径
    :return: 包含已排序画格数据的字典。
    """
    # 这是一个用于分析图像画格并将结果保存到YAML文件中的Python函数。该函数以YAML文件路径作为输入，并在存在YAML文件时加载YAML文件中的数据。然后，它会分析给定图像列表中的每个图像，并将画格的位置和尺寸保存为字符串列表，并将其保存到YAML文件中。
    # 每个图像的分析包括找到画格边缘像素的主要颜色，根据主要颜色计算遮罩，并使用遮罩将画格分成子矩形。然后，函数将每个子矩形的位置和大小保存为字符串列表中的元素。
    # 如果某个图像中有两个或更多的子矩形，则函数可以使用get_marked_frames函数对其进行标记，但是此部分代码目前被注释掉了。
    # 最后，函数将image_data字典按图像文件名排序，并将排序后的字典保存到YAML文件中。
    img_list = get_valid_imgs(img_folder)
    # ================加载YAML文件================
    frame_data = iload_data(frame_yml)

    if image_inds:
        image_list_roi = [img_list[i] for i in image_inds]
        # 从image_data中移除这些图像的数据，以便后面重新生成
        for image in image_list_roi:
            frame_data.pop(image.name, None)
    # ================分析画格================
    if thread_method == 'queue':
        for p, img_file in enumerate(img_list):
            img_file, frame_grid_strs = analyze1frame(img_file, frame_data, auto_subdir, media_type)
            frame_data[img_file.name] = frame_grid_strs
    else:
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            # 提交所有图片处理任务
            futures = [executor.submit(analyze1frame, img_file, frame_data, auto_subdir, media_type) for
                       img_file in img_list]
            for future in as_completed(futures):
                try:
                    img_file, frame_grid_strs = future.result()
                    frame_data[img_file.name] = frame_grid_strs
                except Exception as e:
                    printe(e)

    # 按图像文件名对 frame_data 字典进行排序
    frame_data_sorted = {k: frame_data[k] for k in natsorted(frame_data)}

    # 将 frame_data_sorted 字典保存到 yml 文件
    with open(frame_yml, 'w') as yml_file:
        yaml.dump(frame_data_sorted, yml_file)

    return frame_data_sorted


# @logger.catch
def water_seg(filled_contour, textblocks):
    """
    使用分水岭算法对给定的填充轮廓进行分割，以分离其中的多个对象。

    :param filled_contour: 已填充的轮廓。
    :param textblocks: 文本块。
    :return: 分割后的轮廓列表。
    """
    # 现在我们想要分离图像中的多个对象
    distance = ndi.distance_transform_edt(filled_contour)
    # 创建一个与图像大小相同的标记数组
    markers_full = zeros_like(filled_contour, dtype=int32)

    # 在分水岭算法中，每个标记值代表一个不同的区域，算法会将每个像素点归属到与其距离最近的标记值所代表的区域。
    for g in range(len(textblocks)):
        textblock = textblocks[g]
        markers_full[textblock.br_n, textblock.br_m] = g + 1

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

    if do_dev_pic:
        color_labels_png = current_dir / 'color_labels.png'
        write_pic(color_labels_png, color_labels)

    seg_cnts = []
    for u in range(len(unique_labels)):
        target_label = unique_labels[u]
        binary_img = zeros_like(labels, dtype=uint8)
        binary_img[labels == target_label] = 255
        # binary_img_path = current_dir / f'binary_img{u}.png'
        # write_pic(binary_img_path, binary_img)

        # 查找轮廓
        sep_contours, hierarchy = findContours(binary_img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        for s in range(len(sep_contours)):
            contour = sep_contours[s]
            cnt = Contr(contour)
            # ================对轮廓数值进行初筛================
            condition_p1 = 300 <= cnt.area <= 1000000
            condition_p2 = 80 <= cnt.perimeter <= 50000
            condition_ps = [
                condition_p1,
                condition_p2,
            ]
            print(f'{condition_ps=}, {cnt.area=:.1f}, {cnt.perimeter=:.2f}')
            if all(condition_ps):
                seg_cnts.append(cnt)
    return seg_cnts


# @logger.catch
def get_textblocks(letter_in_contour, media_type, f=None):
    ih, iw = letter_in_contour.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)
    if media_type == 'Comic':
        kh = 1
        kw = kernel_depth
    else:
        kh = kernel_depth
        kw = 1
    kernal_word = kernel_hw(kh, kw)

    # letter_in_contour_inv = bitwise_not(letter_in_contour)
    if do_dev_pic and f is not None:
        letter_in_contour_png = current_dir / f'letter_in_contour_{f}.png'
        write_pic(letter_in_contour_png, letter_in_contour)

    # ================单字轮廓================
    raw_letter_cnts = []
    letter_cnts = []
    if f is None:
        # ================无框================
        letter_contours, letter_hierarchy = findContours(letter_in_contour, RETR_LIST, CHAIN_APPROX_SIMPLE)
        for l in range(len(letter_contours)):
            letter_contour = letter_contours[l]
            letter_cnt = Contr(letter_contour)
            # ================对数值进行初筛================
            condition_c1 = cnt_area_min <= letter_cnt.area <= cnt_area_max
            condition_c2 = brw_min <= letter_cnt.br_w <= brw_max
            condition_c3 = brh_min <= letter_cnt.br_h <= brh_max
            condition_cs = [
                condition_c1,
                condition_c2,
                condition_c3,
            ]
            if all(condition_cs):
                logger.debug(f"{letter_cnt=}")
                raw_letter_cnts.append(letter_cnt)
    else:
        letter_contours, letter_hierarchy = findContours(letter_in_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        for l in range(len(letter_contours)):
            letter_contour = letter_contours[l]
            letter_cnt = Contr(letter_contour)
            raw_letter_cnts.append(letter_cnt)
    # ================再次筛选================
    for l in range(len(raw_letter_cnts)):
        letter_cnt = raw_letter_cnts[l]
        all_letters = drawContours(black_bg.copy(), letter_cnt.contour, -1, 255, FILLED)
        letter_in_word = bitwise_and(letter_in_contour, all_letters)
        px_pts = transpose(nonzero(letter_in_word))
        px_area = px_pts.shape[0]
        if letter_area_min <= px_area <= letter_area_max:
            letter_cnts.append(letter_cnt)
    # logger.debug(f"{len(raw_letter_cnts)=}, {len(letter_contours)=}, {len(letter_cnts)=}")

    # ================单词================
    textwords = []
    letter_cnts4words = deepcopy(letter_cnts)
    # 从左到右，然后从上到下
    letter_cnts4words.sort(key=lambda x: (x.br_x, x.br_y))
    if all_caps:
        word_in_contour = dilate(letter_in_contour, kernal_word, iterations=1)
        word_contours, word_hierarchy = findContours(word_in_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        if do_dev_pic:
            word_in_contour_png = current_dir / f'letter_in_contour_{f}_dilated.png'
            write_pic(word_in_contour_png, word_in_contour)
        word_cnts = []
        for w in range(len(word_contours)):
            word_contour = word_contours[w]
            word_cnt = Contr(word_contour)
            if word_cnt.area >= 50 or len(word_contours) <= 3:
                word_cnts.append(word_cnt)
        for w in range(len(word_cnts)):
            word_cnt = word_cnts[w]
            letter_cnts_in_word = []
            # 枚举每个letter_cnt，检查是否在word_cnt内部
            i = 0
            while i < len(letter_cnts4words):
                letter_cnt = letter_cnts4words[i]
                if letter_cnt.is_inside(word_cnt):
                    letter_cnts_in_word.append(letter_cnt)
                    letter_cnts4words.pop(i)
                else:
                    i += 1
            textword = TextWord(letter_cnts_in_word, media_type)
            textword.add_letter_mask(letter_in_contour)
            textwords.append(textword)
    else:
        while letter_cnts4words:
            # 取最左边的单词，逐词往右查找
            letter_cnt = letter_cnts4words[0]
            letter_cnts4words.remove(letter_cnt)
            textword = TextWord(letter_cnt, media_type)
            while True:
                # 对原始的文本行进行深度复制，以便之后比较是否有变化
                old_textword = deepcopy(textword)
                for letter in letter_cnts4words[:]:
                    # 检查单词的质心和文本行的中心点的高度差异是否在允许的范围内
                    centroid_height_diff = abs(letter.cy - textword.br_center[1])
                    this_height_diff_thres = max_height_diff
                    # 如果单词和文本行的轮廓有交集
                    if textword.ext_brp.intersects(letter.brp):
                        # 且高度差异在允许的范围内
                        if centroid_height_diff <= this_height_diff_thres:
                            # 则认为单词属于这个文本行
                            textword.add_letter_cnt(letter)
                            letter_cnts4words.remove(letter)
                # 如果在这个循环中文本行没有变化（没有新的 letter 被加入），则退出循环
                if len(textword.letter_cnts) == len(old_textword.letter_cnts):
                    break
            textword.add_letter_mask(letter_in_contour)
            # logger.debug(f'{textword.letter_area=:.2f}')
            if textword.letter_area <= letter_area_max:
                textwords.append(textword)
        # ================字母i最上方的点可能被孤立================
        if check_dots:
            dots = [x for x in textwords if x.br_area <= 30 and len(x.letter_cnts) == 1]
            not_dots = [x for x in textwords if x not in dots]
            for d in range(len(dots)):
                dot = dots[d]
                # 筛选出在 dot 下方的 textword，并计算距离
                textwords_under = [textword for textword in not_dots if textword.br_y >= dot.br_y - 30]
                distance_tups = []
                for textword in textwords_under:
                    distance = dot.brp.distance(textword.brp)
                    # 如果textword在dot的上方，将距离乘以3
                    if textword.br_y < dot.br_y:
                        distance *= 3
                    distance_tups.append((distance, textword))
                # 按距离进行排序
                distance_tups.sort(key=lambda x: x[0])
                if distance_tups:
                    closest_dist, closest_textword = distance_tups[0]
                    index = not_dots.index(closest_textword)
                    not_dots[index].add_letter_cnt(dot.letter_cnts[0])
            textwords = not_dots
    textwords.sort(key=lambda x: (x.br_y, x.br_x))

    # ================文本行================
    textlines = []
    textwords4lines = deepcopy(textwords)
    # 从左到右，然后从上到下
    textwords4lines.sort(key=lambda x: (x.br_x, x.br_y))
    while textwords4lines:
        # 取最左边的单词，逐词往右查找
        textword = textwords4lines[0]
        textwords4lines.remove(textword)
        textline = TextLine(textword, media_type)
        while True:
            # 对原始的文本行进行深度复制，以便之后比较是否有变化
            old_textline = deepcopy(textline)
            for textword in textwords4lines[:]:
                # 检查单词的质心和文本行的中心点的高度差异是否在允许的范围内
                centroid_height_diff = abs(textword.br_center[1] - textline.br_center[1])
                this_height_diff_thres = max_height_diff
                # 如果单词和文本行的轮廓有交集
                if textline.ext_brp.intersects(textword.brp):
                    # 且高度差异在允许的范围内
                    if centroid_height_diff <= this_height_diff_thres:
                        # 则认为单词属于这个文本行
                        textline.add_textword(textword)
                        textwords4lines.remove(textword)
            # 如果在这个循环中文本行没有变化（没有新的 word 被加入），则退出循环
            if len(textline.textwords) == len(old_textline.textwords):
                break
        textlines.append(textline)
    textlines.sort(key=lambda x: x.br_y)

    # ================根据文本行检测文本块================
    textblocks = []
    textlines4blocks = deepcopy(textlines)
    # 从上到下然后从左到右排序
    textlines4blocks.sort(key=lambda x: (x.br_y, x.br_x))
    while textlines4blocks:
        # 取最下方的文本行，逐行往上查找
        textline = textlines4blocks[-1]
        textlines4blocks.remove(textline)
        textblock = iTextBlock(textline, media_type)
        while True:
            old_textblock = deepcopy(textblock)
            for textline in textlines4blocks[:]:
                is_inside = textblock.core_brp.intersects(textline.brp) or textline.brp.contains(textblock.core_brp)
                if is_inside:
                    textblock.add_textline(textline)
                    textlines4blocks.remove(textline)
            if len(textblock.textlines) == len(old_textblock.textlines):
                break
        textblocks.append(textblock)
    textblocks.sort(key=lambda x: x.br_y + x.br_x)

    # ================去除相交且相交面积大于等于面积更小的文本块的面积的80%的小文本块================
    i = 0
    while i < len(textblocks):
        tb1 = textblocks[i]
        removed = False
        for j, tb2 in enumerate(textblocks):
            if i != j and tb1.brp.intersects(tb2.brp):
                intersection_area = tb1.brp.intersection(tb2.brp).area
                min_area = min(tb1.br_area, tb2.br_area)
                if intersection_area >= intersect_ratio * min_area:
                    if tb1.br_area < tb2.br_area:
                        textblocks.pop(i)
                        removed = True
                        break
        if not removed:
            i += 1
    # ================去除完全包含在另一个文本块中的小文本块================
    filter_textblocks = []
    for i, tb1 in enumerate(textblocks):
        is_inside_another = False
        for j, tb2 in enumerate(textblocks):
            if i != j and tb2.brp.contains(tb1.brp):
                is_inside_another = True
                break
        if not is_inside_another:
            filter_textblocks.append(tb1)
    textblocks = filter_textblocks
    textblocks = [x for x in textblocks if textblock_letters_min <= x.letter_count <= textblock_letters_max]
    # ================其他处理================
    # if len(textblocks) == 3:
    #     textblocks.sort(key=lambda x: x.br_y)
    return textblocks


# @logger.catch
def get_full_scnts(seg_cnts, filled_contour, filled_contour_split):
    ih, iw = filled_contour.shape[0:2]
    color_img = zeros((ih, iw, 3), dtype=uint8)
    for idx, cnt in enumerate(seg_cnts):
        color = colormap_tab20(idx % 20)[:3]
        color_rgb = tuple(int(c * 255) for c in color)
        color_bgr = color_rgb[::-1]
        drawContours(color_img, [cnt.contour], 0, color_bgr, -1)

    diff_mask = filled_contour - filled_contour_split
    for y, x in argwhere(diff_mask == 255):
        dists = []
        point = Point(x, y)
        for idx, contour in enumerate(seg_cnts):
            if contour.polygon is None:  # 安全检查
                print(f"索引为 {idx} 的轮廓没有polygon属性!")
                continue
            dist = contour.polygon.distance(point)
            dists.append((dist, idx))
        dists.sort()
        # 获取最小距离的索引
        min_dist_idx = dists[0][1]
        # 使用该索引获取颜色
        color = colormap_tab20(min_dist_idx % 20)[:3]
        color_rgb = tuple(int(c * 255) for c in color)
        nearest_color = color_rgb[::-1]
        color_img[y, x] = nearest_color

    new_scnts = []
    for idx in range(len(seg_cnts)):
        color = colormap_tab20(idx % 20)[:3]
        color_rgb = tuple(int(c * 255) for c in color)
        color_bgr = color_rgb[::-1]
        mask = (color_img == color_bgr).all(axis=-1).astype(uint8) * 255
        contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        new_scnts.extend([Contr(contour) for contour in contours])
    seg_cnts = new_scnts
    return seg_cnts


def get_outline_pt(intersection, ref_pt1, ref_pt2, min_dist=1):
    """
    根据交点和参考点获取轮廓点。

    :param intersection: 交点，可以是点、线或多线
    :param ref_pt1: 参考点1
    :param ref_pt2: 参考点2
    :return: 轮廓点
    """
    # 将输入的参考点转换为Point对象
    ref_pt1 = Point(ref_pt1)
    ref_pt2 = Point(ref_pt2)

    potential_pts = []
    if isinstance(intersection, MultiLineString):
        for line in intersection.geoms:
            potential_pts.extend(line.coords)
    elif isinstance(intersection, LineString):
        potential_pts.extend(intersection.coords)
    elif isinstance(intersection, Point):
        potential_pts.append(intersection.coords[0])

    # 排除离ref_point1和ref_point2距离小于等于min_dist的点
    potential_pts = [pt for pt in potential_pts if
                     ref_pt1.distance(Point(pt)) > min_dist and ref_pt2.distance(Point(pt)) > min_dist]

    if not potential_pts:
        return None

    # 找到距离ref_point2最近的点
    closest_pt = min(potential_pts, key=lambda pt: ref_pt2.distance(Point(pt)))
    return Point(closest_pt)


@logger.catch
def pivot_proc(filter_cnt, filled_contour, letter_in_contour, textblocks, f):
    filled_contour_split = filled_contour.copy()

    # 创建一个预览图像，用于显示分割效果
    preview_canvas = cvtColor(filled_contour.copy(), COLOR_GRAY2BGR)
    # 将轮廓中的字母部分设置为黑色，以便在示意图中清晰地看到分割线
    preview_canvas[letter_in_contour == 255] = 0
    for textblock in textblocks:
        # 在示意图上标记文本块的中心点，并绘制文本块的轮廓
        preview_canvas = circle(preview_canvas, pt2tup(textblock.block_poly.centroid), 5, color_green, -1)
        preview_canvas = drawContours(preview_canvas, [textblock.block_contour], 0, color_olive, 1)

    split_preview = deepcopy(preview_canvas)
    for i in range(len(textblocks) - 1):
        textblock_comb = textblocks[i:i + 2]
        # 使用针转切割
        textblock1, textblock2 = textblock_comb[0:2]
        intersect_pts = nearest_points(textblock1.block_poly, textblock2.block_poly)
        intersect_points_geom = MultiPoint(intersect_pts)
        # 计算最近距离交点的中心（质心）
        dist_center = intersect_points_geom.centroid
        # 定义连线
        link_line = LineString([textblock1.br_center, textblock2.br_center])
        sect1 = textblock1.block_poly.convex_hull.intersection(link_line)
        sect2 = textblock2.block_poly.convex_hull.intersection(link_line)
        outline_pt1 = get_outline_pt(sect1, textblock1.br_center, textblock2.br_center)
        outline_pt2 = get_outline_pt(sect2, textblock2.br_center, textblock1.br_center)
        if outline_pt1 is None or outline_pt2 is None:
            img = filled_contour.copy()
            blended_img = get_textblock_bubbles(img, textblocks)
            check_pts_png = current_dir / 'check_pts.png'
            write_pic(check_pts_png, blended_img)
        if outline_pt1 is None and outline_pt2 is None:
            pt_center = None
        elif outline_pt1 is None:
            pt_center = outline_pt2
        elif outline_pt2 is None:
            pt_center = outline_pt1
        else:
            points = MultiPoint([outline_pt1, outline_pt2])
            mid_point = points.centroid
            pt_center = mid_point

        if pt_center is None:
            raise ValueError("pt_center是None，无法继续。")

        logger.info(f"{outline_pt1}, {outline_pt2}, {pt_center=}")
        # 在质心周围创建一个像素范围
        pixel_range = range(-px_step, px_step + 1)

        # ================计算合适的分割线长度================
        line_length = 500
        lengths = []
        for angle in range(0, 180, angle_step):
            rad_angle = radians(angle)
            # 计算线段的两个端点
            line_pt1 = (
                pt_center.x + line_length * cos(rad_angle),
                pt_center.y + line_length * sin(rad_angle)
            )
            line_pt2 = (
                pt_center.x - line_length * cos(rad_angle),
                pt_center.y - line_length * sin(rad_angle)
            )
            cline = LineString([line_pt1, line_pt2])
            intersection = cline.intersection(filter_cnt.polygon)
            if intersection.geom_type == "LineString":
                lengths.append(intersection.length)
        min_intersect_length = min(lengths)
        seg_line_len = min_intersect_length * 1.5
        seg_line_len = max(seg_line_min, min(seg_line_max, seg_line_len))
        if not filter_cnt.polygon.contains(pt_center):
            seg_line_len = (seg_line_min + seg_line_max) / 2

        # ================计算合适的分割================
        all_tups = []
        for p in pixel_range:
            # 创建一个新的质心点，偏移了i个像素
            cur_pt_center = Point(pt_center.x, pt_center.y + p)
            for angle in range(0, 180, angle_step):
                rad_angle = radians(angle)
                # 计算线段的两个端点
                sep_line_pt1 = (
                    cur_pt_center.x + seg_line_len * cos(rad_angle),
                    cur_pt_center.y + seg_line_len * sin(rad_angle)
                )
                sep_line_pt2 = (
                    cur_pt_center.x - seg_line_len * cos(rad_angle),
                    cur_pt_center.y - seg_line_len * sin(rad_angle)
                )
                sep_line_pts = [sep_line_pt1, sep_line_pt2]
                sep_line = LineString(sep_line_pts)

                # 计算分隔线与过滤后轮廓的交点
                cnt_line_sect = sep_line.intersection(filter_cnt.polygon)
                # 不会改变原始图形的形状，但会清除任何存在的几何不规则性，例如自交叉的线条或重叠的点。
                tb1_sect = sep_line.intersection(textblock1.block_poly.buffer(0))
                tb2_sect = sep_line.intersection(textblock2.block_poly.buffer(0))
                comb_len = cnt_line_sect.length + 1000 * (tb1_sect.length + tb2_sect.length)
                tup = (cur_pt_center, angle, sep_line, cnt_line_sect, tb1_sect, tb2_sect, comb_len)
                all_tups.append(tup)

        all_tups.sort(key=lambda x: x[-1])
        best_tup = all_tups[0]
        cur_pt_center, angle, sep_line, cnt_line_sect, tb1_sect, tb2_sect, comb_len = best_tup
        if cnt_line_sect.geom_type == "LineString":
            coords_list = list(cnt_line_sect.coords)
        else:  # cnt_line_sect.geom_type == "MultiLineString":
            dists = []
            # 计算每个线段到pt_center的距离
            for linestring in cnt_line_sect.geoms:
                distance = linestring.distance(pt_center)
                dists.append(distance)
            # 寻找最小距离及其对应的线段
            min_dist = min(dists)
            # 获取最小距离的索引
            index_of_closest = dists.index(min_dist)
            closest_linestring = cnt_line_sect.geoms[index_of_closest]
            # 从最近的线段中获取坐标列表
            coords_list = list(closest_linestring.coords)

        if len(coords_list) >= 2:
            # 根据坐标列表获取起始点和终点
            p0, p1 = map(lambda coord: tuple(map(floor, coord)), coords_list[:2])
        else:
            logger.error(f'{coords_list=}')
            # 生成示意图
            error_preview = deepcopy(preview_canvas)
            # 保存示意图
            error_preview_png = current_dir / 'error_preview.png'
            write_pic(error_preview_png, error_preview)

        logger.debug(f'{cur_pt_center=}, {p0=}, {p1=}, {outline_pt1=}, {outline_pt2=}')
        filled_contour_split = line(filled_contour_split, p0, p1, 0, 2)

        ct1 = pt2tup(textblock_comb[0].block_poly.centroid)
        ct2 = pt2tup(textblock_comb[1].block_poly.centroid)
        # 在示意图上绘制从两个文本块中心点连接的线条
        split_preview = line(split_preview, ct1, ct2, color_maroon, 2)
        # 在示意图上绘制实际的分割线
        split_preview = line(split_preview, p0, p1, color_red, 2)
        if outline_pt1 is not None and outline_pt2 is not None:
            split_preview = line(split_preview, pt2tup(outline_pt1), pt2tup(outline_pt2), color_slate_blue, 1)
        # 在示意图上标记分割线的中心点
        split_preview = circle(split_preview, pt2tup(pt_center), 3, color_yellow, -1)
        split_preview = circle(split_preview, pt2tup(cur_pt_center), 3, color_blue, -1)

    contours, _ = findContours(filled_contour_split, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    seg_cnts = [Contr(contour) for contour in contours if Contr(contour).area >= area_min]
    seg_cnts = get_full_scnts(seg_cnts, filled_contour, filled_contour_split)

    if do_dev_pic:
        split_preview_png = current_dir / f'split_preview_{f}.png'
        filled_contour_split_png = current_dir / 'filled_contour_split.png'
        write_pic(split_preview_png, split_preview)
        write_pic(filled_contour_split_png, filled_contour_split)
    return seg_cnts, True


@logger.catch
def seg_bubbles(filter_cnts, bubble_mask, letter_mask, media_type):
    ih, iw = bubble_mask.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)
    single_cnts = []

    all_textblocks = []
    for f in range(len(filter_cnts)):
        # ================每个初始对话气泡================
        filter_cnt = filter_cnts[f]
        br_area_real = (filter_cnt.br_w - 2) * (filter_cnt.br_h - 2)
        fulfill_ratio = filter_cnt.area / br_area_real

        filled_contour = drawContours(black_bg.copy(), [filter_cnt.contour], 0, 255, -1)
        # bubble_in_contour = bitwise_and(bubble_mask, filled_contour)
        letter_in_contour = bitwise_and(letter_mask, filled_contour)

        raw_textblocks = get_textblocks(letter_in_contour, media_type, f)
        # ================排除过小的文本块================
        big_textblocks = [x for x in raw_textblocks if textblock_area_min <= x.block_cnt.area <= textblock_area_max]
        small_textblocks = [x for x in raw_textblocks if x.block_cnt.area < textblock_area_min]
        note_textblocks = []
        if big_textblocks:
            max_block_cnt_area = max(x.block_cnt.area for x in big_textblocks)
            logger.warning(f'{max_block_cnt_area=}')
            # ================检查音符================
            dists_dic = {}
            if check_note and len(big_textblocks) >= 2 and max_block_cnt_area >= note_area_max:
                for idx, textblock in enumerate(big_textblocks[:]):
                    # if note_area_min <= textblock.block_cnt.area <= note_area_max:
                    dists = [filter_cnt.polygon.boundary.distance(Point(p)) for p in
                             textblock.block_poly.exterior.coords]
                    min_dist = min(dists)
                    dists_dic[idx] = dists
                    logger.warning(f'[{idx}]{textblock.block_cnt.area=}, {min_dist=}')
                    if min_dist <= max_note_dist:
                        note_textblocks.append(textblock)
                        big_textblocks.remove(textblock)
                if dists_dic:
                    print()
            textblocks = big_textblocks
        else:
            textblocks = raw_textblocks
        all_textblocks.extend(textblocks)

        # ================如果有音符================
        if note_textblocks:
            for textblock in note_textblocks:
                textblock_contour_mask = zeros(letter_mask.shape, dtype=uint8)
                drawContours(textblock_contour_mask, [textblock.block_contour], 0, 255, -1)
                letter_in_textblock = bitwise_and(letter_mask, textblock_contour_mask)
                contours, _ = findContours(letter_in_textblock, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    # 找到从音符轮廓到边界的最短线段
                    dists = [(Point(p1[0]).distance(Point(p2[0])), (p1[0], p2[0])) for p1 in cnt for p2 in
                             filter_cnt.contour]
                    dists.sort(key=lambda x: x[0])
                    min_segment = dists[0][1]
                    # 音符轮廓及最短线段
                    mask = zeros(letter_mask.shape, dtype=uint8)
                    drawContours(mask, [cnt], 0, 255, -1)
                    line(mask, tuple(min_segment[0]), tuple(min_segment[1]), 255, 2)
                    dilated_mask = dilate(mask, kernel3, iterations=1)
                    filled_contour[dilated_mask == 255] = 0

            if do_dev_pic:
                filled_contour_preview_png = current_dir / 'filled_contour_preview.png'
                write_pic(filled_contour_preview_png, filled_contour)

            contours_in_filled_contour, _ = findContours(filled_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours_in_filled_contour, key=contourArea)
            filter_cnt = Contr(largest_contour)

        # ================不是矩形且有两个以上文本框================
        if len(textblocks) >= 2 and fulfill_ratio <= 0.95:
            # 有2个以上的文字块轮廓则进行切割
            # 先用分水岭切割
            for t in range(len(textblocks)):
                textblock = textblocks[t]
                logger.debug(f'[{t + 1}]{textblock=}')
            seg_cnts = water_seg(filled_contour, textblocks)

            # 先检查数量一致性
            seg_success = (len(seg_cnts) == len(textblocks))
            # 再检查分割是否不破坏文本
            if seg_success and check_intact:
                for cnt in seg_cnts:
                    for textblock in textblocks:
                        if not cnt.polygon.is_valid:
                            cnt.polygon = cnt.polygon.buffer(0)
                        if not textblock.block_poly.is_valid:
                            textblock.block_poly = textblock.block_poly.buffer(0)
                        intersection = cnt.polygon.intersection(textblock.block_poly)
                        if intersection.area not in [0, textblock.block_poly.area]:
                            seg_success = False
                            break
                    if not seg_success:
                        break

            if not seg_success and use_pivot_split:
                if do_dev_pic:
                    textblock_bubbles = get_textblock_bubbles(bubble_mask, textblocks)
                    cp_textblock_jpg = current_dir / f'TextBlock-{f}.jpg'
                    write_pic(cp_textblock_jpg, textblock_bubbles)
                logger.warning(f'{len(textblocks)=}, 分水岭切割失败')
                # 再用轴点针转切割
                seg_cnts, seg_success = pivot_proc(filter_cnt, filled_contour, letter_in_contour, textblocks, f)
            if not seg_success:
                logger.error(f'{len(textblocks)=}, 轴点针转切割失败')
                seg_cnts = [filter_cnt]
            single_cnts.extend(seg_cnts)
        else:
            single_cnts.append(filter_cnt)
    return single_cnts, all_textblocks


def get_bubbles_by_cp(img_file, color_pattern, frame_grid_strs, CTD_mask, media_type, auto_subdir):
    """
    分析气泡并根据指定的颜色模式对其进行处理。

    :param img_file: 输入的漫画或其他带文字的图像。
    :param color_pattern: 用于分析和处理气泡的颜色模式。
    :param frame_grid_strs: 框架网格字符串列表，用于确定气泡在图像中的位置和顺序。
    :param CTD_mask: 漫画文本检测器生成的气泡掩码。
    """

    added_frames_jpg = img_file.parent / f'{img_file.stem}-加框.jpg'
    ctd_png = auto_subdir / f'{img_file.stem}-CTD.png'
    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    if added_frames_jpg.exists():
        image_raw = imdecode(fromfile(added_frames_jpg, dtype=uint8), -1)
    image_raw = toBGR(image_raw)
    ih, iw = image_raw.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)

    cp_bubble, cp_letter = color_pattern
    logger.debug(f'{color_pattern=}, {cp_bubble=}, {cp_letter=}')
    if cp_bubble == '':
        color_bubble = None
    elif isinstance(cp_bubble, list):
        # 气泡为渐变色
        color_bubble = ColorGradient(cp_bubble)
    else:
        # 气泡为单色
        color_bubble = Color(cp_bubble)

    if isinstance(cp_letter, list):
        # 文字为双色
        color_letter = ColorDouble(cp_letter)
    else:
        # 文字为单色
        color_letter = Color(cp_letter)
    letter_mask = color_letter.get_range_img(image_raw)

    if color_bubble is None:
        # ================排除已经识别的气泡================
        mask_pics = [x for x in all_masks if x.stem.startswith(img_file.stem)]
        if mask_pics:
            single_cnts = get_single_cnts(image_raw, mask_pics)
            bubble_contours = [x.contour for x in single_cnts]
            ready_bubble_mask = drawContours(black_bg.copy(), bubble_contours, -1, 255, FILLED)
            dilated_mask = dilate(ready_bubble_mask, kernel5, iterations=1)
            dilated_mask_inv = bitwise_not(dilated_mask)
            letter_mask = bitwise_and(letter_mask, dilated_mask_inv)
        if use_torch and CTD_model is not None:
            CTD_mask = get_CTD_mask(image_raw)
            # 将comictextdetector_mask中大于或等于127的部分设置为255，其余部分设置为0
            ret, CTD_mask = threshold(CTD_mask, 127, 255, THRESH_BINARY)
            letter_mask = bitwise_and(letter_mask, CTD_mask)

        all_textblocks = get_textblocks(letter_mask, media_type)
        ordered_cnts = [x.expanded_cnt for x in all_textblocks]

        cp_raw_jpg = auto_subdir / f'{img_file.stem}-Raw-{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_textblock_jpg = auto_subdir / f'{img_file.stem}-TextBlock-{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_preview_jpg = auto_subdir / f'{img_file.stem}-{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_mask_cnt_png = auto_subdir / f'{img_file.stem}-Mask-{color_letter.rgb_str}-{color_letter.color_name}.png'
    else:
        if color_bubble.type == 'Color':
            bubble_mask = color_bubble.get_range_img(image_raw)
            left_sample, right_sample = None, None
        else:  # ColorGradient
            img_tuple = color_bubble.get_range_img(image_raw)
            bubble_mask, left_sample, right_sample = img_tuple

        if do_dev_pic:
            bubble_mask_png = current_dir / 'bubble_mask.png'
            write_pic(bubble_mask_png, bubble_mask)

        cp_raw_jpg = auto_subdir / f'{img_file.stem}-Raw-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_textblock_jpg = auto_subdir / f'{img_file.stem}-TextBlock-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_preview_jpg = auto_subdir / f'{img_file.stem}-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.jpg'
        cp_mask_cnt_png = auto_subdir / f'{img_file.stem}-Mask-{color_bubble.rgb_str}-{color_bubble.color_name}~{color_letter.rgb_str}-{color_letter.color_name}.png'

        filter_cnts = get_raw_bubbles(bubble_mask, letter_mask, left_sample, right_sample, CTD_mask)
        colorful_raw_bubbles = get_colorful_bubbles(image_raw, filter_cnts)
        # ================切割相连气泡================
        single_cnts, all_textblocks = seg_bubbles(filter_cnts, bubble_mask, letter_mask, media_type)
        # ================通过画格重新排序气泡框架================
        single_cnts_grids = []
        for f in range(len(frame_grid_strs)):
            frame_grid_str = frame_grid_strs[f]
            # 使用正则表达式提取字符串中的所有整数
            int_values = list(map(int, findall(r'\d+', frame_grid_str)))
            # 按顺序分配值
            x, y, w, h, xx, yy, ww, hh = int_values
            single_cnts_grid = []
            for s in range(len(single_cnts)):
                single_cnt = single_cnts[s]
                if x <= single_cnt.cx <= x + w and y <= single_cnt.cy <= y + h:
                    single_cnts_grid.append(single_cnt)
                else:
                    # 计算该single_cnt到所有frame_grid的距离
                    dists = []
                    for frame_grid_str_opt in frame_grid_strs:
                        int_values_opt = list(map(int, findall(r'\d+', frame_grid_str_opt)))
                        x_opt, y_opt, w_opt, h_opt, _, _, _, _ = int_values_opt
                        rect_opt = (x_opt, y_opt, w_opt, h_opt)
                        dist2rect = get_dist2rect(single_cnt.polygon.centroid, rect_opt)
                        dists.append(dist2rect)

                    # 寻找最小距离及其对应的frame_grid的索引
                    closest_dist = min(dists)
                    closest_frame_idx = dists.index(closest_dist)
                    if closest_frame_idx == f:
                        single_cnts_grid.append(single_cnt)

            if media_type == 'Manga':
                ax = -1
            else:
                ax = 1
            single_cnts_grid.sort(key=lambda x: ax * x.cx + x.cy)
            single_cnts_grids.append(single_cnts_grid)
        ordered_cnts = list(chain(*single_cnts_grids))

    if do_dev_pic:
        letter_mask_png = current_dir / 'letter_mask.png'
        write_pic(letter_mask_png, letter_mask)
    if CTD_mask is not None:
        write_pic(ctd_png, CTD_mask)
    if len(ordered_cnts) >= 1:
        colorful_single_bubbles = get_colorful_bubbles(image_raw, ordered_cnts)
        textblock_bubbles = get_textblock_bubbles(image_raw, all_textblocks)

        # 创建一个带有bg_alpha透明度原图背景的图像
        transparent_img = zeros((ih, iw, 4), dtype=uint8)
        transparent_img[..., :3] = image_raw
        transparent_img[..., 3] = int(255 * bg_alpha)
        # 在透明图像上绘制contours，每个contour使用不同颜色

        for s in range(len(ordered_cnts)):
            bubble_cnt = ordered_cnts[s]
            # 从 tab20 颜色映射中选择一个颜色
            color = colormap_tab20(s % 20)[:3]
            color_rgb = tuple(int(c * 255) for c in color)
            color_bgra = color_rgb[::-1] + (255,)
            drawContours(transparent_img, [bubble_cnt.contour], -1, color_bgra, -1)

        # write_pic(cp_raw_jpg, colorful_raw_bubbles)
        write_pic(cp_textblock_jpg, textblock_bubbles)
        write_pic(cp_preview_jpg, colorful_single_bubbles)
        write_pic(cp_mask_cnt_png, transparent_img)
    else:
        transparent_img = None
    return cp_mask_cnt_png


# @logger.catch
def analyze1pic(img_file, frame_data, color_patterns, media_type, auto_subdir):
    logger.warning(f'{img_file=}')
    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    image_raw = toBGR(image_raw)
    ih, iw = image_raw.shape[0:2]
    # ================矩形画格信息================
    frame_grid_strs = frame_data.get(img_file.name, [f'0,0,{iw},{ih}~0,0,{iw},{ih}'])
    # ================模型检测文字，文字显示为白色================
    if use_torch and CTD_model is not None:
        CTD_mask = get_CTD_mask(image_raw)
    else:
        CTD_mask = None
    # ================针对每一张图================
    for c in range(len(color_patterns)):
        # ================遍历每种气泡文字颜色组合================
        color_pattern = color_patterns[c]
        cp_mask_cnt_png = get_bubbles_by_cp(img_file, color_pattern, frame_grid_strs, CTD_mask, media_type, auto_subdir)


@timer_decorator
# @logger.catch
def step1_analyze_bubbles(img_folder, media_type, auto_subdir):
    """
    分析气泡，获取气泡位置和尺寸，将结果可视化并保存到文件中。
    """
    # 这是一个用于分析漫画气泡并将结果保存到文件中的Python函数。该函数首先从YAML文件中加载画格信息，然后对给定的漫画图像列表进行处理。
    # 函数的主要工作是遍历每种气泡文字颜色组合，并分析每张图像中的气泡和文字。对于每种颜色组合，函数使用该颜色来创建气泡蒙版，并使用白色文字蒙版来检测文本。函数使用这些蒙版来过滤出符合条件的气泡，并对相邻的气泡进行切割和排序，以获得正确的气泡框架。
    # 接下来，函数为每个气泡框架绘制一个不同的颜色，并将结果保存为JPEG文件和PNG文件。这些文件用于可视化结果和后续的处理。
    # 最后，函数将处理后的气泡蒙版复制到原始图像所在的文件夹中。
    img_list = get_valid_imgs(img_folder)
    frame_yml = img_folder.parent / f'{img_folder.name}.yml'
    frame_data = iload_data(frame_yml)
    all_masks_old = get_valid_imgs(img_folder, mode='mask')

    if pic_thread_method == 'queue':
        for p, img_file in enumerate(img_list):
            analyze1pic(img_file, frame_data, color_patterns, media_type, auto_subdir)
    else:
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            futures = [
                executor.submit(analyze1pic, img_file, frame_data, color_patterns, media_type, auto_subdir) for
                img_file in img_list]

    # ================搬运气泡蒙版================
    auto_all_masks = get_valid_imgs(auto_subdir, mode='mask')
    # 如果步骤开始前没有气泡蒙版
    if not all_masks_old:
        for mask_src in auto_all_masks:
            mask_dst = img_folder / mask_src.name
            copy2(mask_src, mask_dst)
    return auto_all_masks


# @logger.catch
# @timer_decorator
def get_single_cnts(image_raw, mask_pics):
    """
    从原始图像和对应的掩码图像中提取单个气泡的轮廓及其裁剪后的图像。

    :param image_raw: 原始图像，通常为漫画或其他带文字的图像。
    :param mask_pics: 包含掩码图像的列表，这些掩码用于在原始图像中找到气泡。
    :return 单个气泡轮廓及其裁剪后的图像的列表
    """
    ih, iw = image_raw.shape[0:2]
    black_bg = zeros((ih, iw), dtype=uint8)
    single_cnts = []

    # 一次性读取所有的透明图像
    transparent_imgs = [imdecode(fromfile(mask_pic, dtype=uint8), -1) for mask_pic in mask_pics]

    for m, transparent_img in enumerate(transparent_imgs):
        # ================获取轮廓================
        # 将半透明像素变为透明
        alpha_channel = transparent_img[..., 3]
        transparent_mask = alpha_channel < 255
        opaque_mask = alpha_channel == 255
        transparent_img[transparent_mask] = 0
        # 获取所有完全不透明像素的颜色
        non_transparent_pxs = transparent_img[opaque_mask, :3]
        unique_colors = np.unique(non_transparent_pxs, axis=0)

        # 对于每种颜色，提取对应的contour
        contour_list = []
        for color in unique_colors:
            mask = np.all(transparent_img[..., :3] == color, axis=-1)
            mask = (mask & opaque_mask).astype(uint8) * 255
            contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            contour_list.extend(contours)

        # ================遍历每种气泡文字颜色组合================
        color_pattern = get_color_pattern(mask_pics[m])
        cp_bubble, cp_letter = color_pattern

        if cp_bubble == '':
            color_bubble = None
        elif isinstance(cp_bubble, list):
            # 气泡为渐变色
            color_bubble = ColorGradient(cp_bubble)
            img_tuple = color_bubble.get_range_img(image_raw)
            bubble_mask, left_sample, right_sample = img_tuple
        else:
            # 气泡为单色
            color_bubble = Color(cp_bubble)
            bubble_mask = color_bubble.get_range_img(image_raw)

        if isinstance(cp_letter, list):
            # 文字为双色
            color_letter = ColorDouble(cp_letter)
        else:
            # 文字为单色
            color_letter = Color(cp_letter)
        letter_mask = color_letter.get_range_img(image_raw)

        # ================获取对应的气泡文字蒙版================
        for c, contour in enumerate(contour_list):
            single_cnt = Contr(contour)
            if single_cnt.area >= 30:
                # 在黑色背景上绘制单个轮廓的白色填充图像
                bit_white_bubble = drawContours(black_bg.copy(), [single_cnt.contour], 0, 255, -1)
                bit_white_bubble_inv = bitwise_not(bit_white_bubble)

                # 获取原始图像在bit_white_bubble范围内的图像，其他部分为气泡颜色
                image_bubble_only = bitwise_and(image_raw, image_raw, mask=bit_white_bubble)
                if color_bubble is not None:
                    image_bubble_only[bit_white_bubble == 0] = color_bubble.rgb
                image_bubble_only_inv = bitwise_not(image_bubble_only)

                # 获取气泡内的文字
                letter_in_bubble = bitwise_and(letter_mask, letter_mask, mask=bit_white_bubble)
                letter_in_bubble_inv = bitwise_not(letter_in_bubble)

                # 获取所有非零像素的坐标
                px_pts = transpose(nonzero(letter_in_bubble))
                # 计算非零像素的数量
                px_area = px_pts.shape[0]
                # 获取所有非零像素的x和y坐标
                all_x = px_pts[:, 1]
                all_y = px_pts[:, 0]
                logger.debug(f'{px_area=}')

                if px_area >= 3:
                    # 计算不带padding的最小和最大x、y坐标
                    raw_min_x, raw_max_x = np.min(all_x), np.max(all_x)
                    raw_min_y, raw_max_y = np.min(all_y), np.max(all_y)

                    # 计算中心点坐标
                    center_x = (raw_min_x + raw_max_x) / 2
                    center_y = (raw_min_y + raw_max_y) / 2
                    center_pt = (int(center_x), int(center_y))

                    # 添加指定的padding
                    min_x, max_x = raw_min_x - padding, raw_max_x + padding
                    min_y, max_y = raw_min_y - padding, raw_max_y + padding
                    # 限制padding后的坐标范围不超过原始图像的边界
                    min_x, min_y = max(min_x, 0), max(min_y, 0)
                    max_x, max_y = min(max_x, iw), min(max_y, ih)
                    letter_coors = (raw_min_x, raw_min_y, raw_max_x, raw_max_y, min_x, min_y, max_x, max_y, center_pt)

                    if isinstance(cp_bubble, list):
                        mod_img = letter_in_bubble_inv.copy()
                    elif color_bubble is None:
                        if color_letter.rgb_str == 'ffffff':
                            # 文字颜色为白色
                            mod_img = letter_in_bubble_inv.copy()
                        elif color_letter.rgb_str == '000000':
                            # 文字颜色为黑色
                            mod_img = image_bubble_only.copy()
                        else:
                            mod_img = letter_in_bubble_inv.copy()
                    else:
                        # 从带有白色背景的图像中裁剪出带有padding的矩形区域
                        if color_bubble.rgb_str == 'ffffff':
                            # 气泡颜色为白色则使用原图
                            mod_img = image_bubble_only.copy()
                        elif color_bubble.rgb_str == '000000':
                            # 气泡颜色为黑色则使用原图反色
                            mod_img = image_bubble_only_inv.copy()
                        elif color_letter.rgb_str == 'ffffff':
                            # 文字颜色为白色则使用原图反色把气泡颜色替换为白色后的转换成的灰度图
                            mod_img = image_bubble_only_inv.copy()
                            mod_img[bubble_mask == 255] = color_white
                            mod_img[bit_white_bubble_inv == 255] = color_white
                            if do_dev_pic:
                                mod_img_png = current_dir / 'mod_img.png'
                                write_pic(mod_img_png, mod_img)
                        elif color_letter.rgb_str == '000000':
                            # 文字颜色为黑色则使用原图把气泡颜色替换为白色后的转换成的灰度图
                            mod_img = image_bubble_only.copy()
                            mod_img[bubble_mask == 255] = color_white
                            mod_img[bit_white_bubble_inv == 255] = color_white
                        else:
                            mod_img = letter_in_bubble_inv.copy()

                    if len(mod_img.shape) == 3:
                        mod_img = cvtColor(mod_img, COLOR_BGR2GRAY)
                    if len(mod_img.shape) == 2:
                        mod_img = cvtColor(mod_img, COLOR_GRAY2BGR)
                    cropped_img = mod_img[min_y:max_y, min_x:max_x]

                    single_cnt.add_cropped_img(cropped_img, letter_coors, color_pattern)
                    single_cnts.append(single_cnt)
    return single_cnts


# @logger.catch
def get_color_pattern(mask_pic):
    mask_prefix, par, cp_str = mask_pic.stem.partition('-Mask-')
    # 先切分气泡颜色和文字颜色
    # ffffff-15-white~000000-60-black
    # 75a1ba-c3d7d8-cadetblue-lightgray~000000-black
    a_str, partition, b_str = cp_str.partition('~')
    a_parts = a_str.split('-')
    if b_str == '':
        color_pattern = [
            '',
            f'{a_parts[0]}-{a_parts[1]}',
        ]
    else:
        b_parts = b_str.split('-')
        if len(a_parts) == 4:
            color_pattern = [
                [
                    f'{a_parts[0]}-{a_parts[2]}',
                    f'{a_parts[1]}-{a_parts[3]}',
                ],
                f'{b_parts[0]}-{b_parts[1]}',
            ]
        else:
            color_pattern = [
                f'{a_parts[0]}-{a_parts[1]}',
                f'{b_parts[0]}-{b_parts[1]}',
            ]
    return color_pattern


def get_start_panels(frame_grid_strs):
    panels = []
    for f in range(len(frame_grid_strs)):
        frame_grid_str = frame_grid_strs[f]
        # 使用正则表达式提取字符串中的所有整数
        int_values = list(map(int, findall(r'\d+', frame_grid_str)))
        if len(int_values) == 8:
            # 按顺序分配值
            x, y, w, h, xx, yy, ww, hh = int_values
            outer_br = (x, y, w, h)
            inner_br = (xx, yy, ww, hh)
            br_tup = (outer_br, inner_br)
            panels.append(br_tup)
    return panels


def get_panel_layers(pic_div_psd):
    panel_layers = []
    if pic_div_psd.exists():
        psd = PSDImage.open(pic_div_psd)
        layers = [layer for layer in psd]
        # ================读取 PSD 文件中所有可见的非背景图层================
        visible_layers = [layer for layer in layers if layer.visible]
        visible_layers = [layer for layer in visible_layers if layer.name != '背景']
        for n in range(len(visible_layers)):
            # ================每个图层================
            layer = visible_layers[n]
            layer_name = layer.name
            if layer_name.startswith('图层'):
                layer_name = layer_name.removeprefix('图层').strip()
            # 当前图层名称为数字时，计入画格图层
            if layer_name.isdigit():
                panel_layers.append(layer)
    return panel_layers


# @logger.catch
@timer_decorator
def get_grid_masks(img_file, frame_grid_strs):
    """
    根据给定的矩形框架字符串列表和 PSD 文件，生成每个框架的二值掩膜。

    :param img_file: 输入的漫画或其他带文字的图像。
    :param frame_grid_strs: 字符串列表，包含每个矩形框架的位置、宽度和高度信息。

    :return: 掩膜图像列表，包含每个框架的二值掩膜。
    """

    img_folder = img_file.parent
    auto_subdir = Auto / img_folder.name
    pic_div_psd = img_folder / f'{img_file.stem}-分框.psd'
    marked_frames_jpg = auto_subdir / f'{img_file.stem}-画格.jpg'

    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    panel_pil = fromarray(cvtColor(image_raw, COLOR_BGR2RGB))
    panel_draw = ImageDraw.Draw(panel_pil, 'RGBA')

    ih, iw = image_raw.shape[0:2]
    black_bg = zeros((ih, iw), uint8)
    logger.warning(f'{frame_grid_strs=}')

    panels = get_start_panels(frame_grid_strs)
    # ================读取psd画格================
    panel_layers = get_panel_layers(pic_div_psd)
    # ================按数字序数排序================
    panel_layers.sort(key=lambda x: int(x.name.removeprefix('图层').strip()))
    # ================以正确顺序插入到panels================
    for p in range(len(panel_layers)):
        # ================每个画格图层================
        layer = panel_layers[p]
        layer_index = int(layer.name.removeprefix('图层').strip())
        panels.insert(layer_index - 1, layer)

    # ================生成grid_masks================
    grid_masks = []
    for p in range(len(panels)):
        # ================每个画格图层================
        panel = panels[p]
        # 从 tab20 颜色映射中选择一个颜色
        color = colormap_tab20(p % 20)[:3]
        color_rgb = tuple(int(c * 255) for c in color)
        color_bgra = color_rgb[::-1] + (255,)

        if isinstance(panel, tuple):
            outer_br, inner_br = panel
            logger.warning(f'{inner_br=}')
            x, y, w, h = outer_br
            xx, yy, ww, hh = inner_br
            # ================绘制外圈矩形================
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            # ================内圈矩形================
            pt3 = (xx, yy)
            pt4 = (xx + ww, yy + hh)
            panel_draw.rectangle([pt1, pt2], outline=color_bgra, width=3)
            # 以外圈矩形为基点计算序数坐标
            text_pos = (x + 5, y - 5)
            # 生成矩形画格信息对应的二值图像，画格内部为 255（白色），其他部分为 0（黑色）
            filled_frame_outer = rectangle(black_bg.copy(), pt1, pt2, 255, -1)
            filled_frame_inner = rectangle(black_bg.copy(), pt3, pt4, 255, -1)
            grid_masks.append(filled_frame_outer)
        else:
            layer = panel
            layer_index = int(layer.name.removeprefix('图层').strip())

            # 获取图层的边界框坐标
            lef, topp, righ, bott = layer.bbox
            # ================numpy================
            layer_img_np = layer.numpy()
            # 将图层数据转换为 0-255 范围的 uint8 类型
            layer_img_np = (layer_img_np * 255).astype(uint8)

            # 对图层的 alpha 通道应用阈值处理，生成二值化掩膜
            ret, layer_mask_raw = threshold(layer_img_np[:, :, 3], 0, 255, THRESH_BINARY)

            layer_mask = fromarray(black_bg.copy())
            layer_mask_raw = fromarray(layer_mask_raw)
            # 将二值化掩膜粘贴到黑色背景上，以生成完整的掩膜图像
            layer_mask.paste(layer_mask_raw, (lef, topp))
            # 将图像对象转换为 numpy 数组
            layer_mask = asarray(layer_mask)

            px_pts = transpose(nonzero(layer_mask))
            px_area = px_pts.shape[0]
            all_x = px_pts[:, 1]
            all_y = px_pts[:, 0]
            all_x.sort()
            all_y.sort()

            # 计算图层边界框的位置、宽度和高度
            layer_x = min(all_x)
            layer_y = min(all_y)
            layer_w = max(all_x) - min(all_x)
            layer_h = max(all_y) - min(all_y)
            layer_br = (layer_x, layer_y, layer_w, layer_h)
            logger.warning(f'{layer_br=}')
            layer_area = layer_w * layer_h
            # 像素面积与图层边界框面积之比
            px_br_ratio = px_area / layer_area
            if px_br_ratio >= 1.2:
                # 使用黑色背景创建填充的矩形框
                pt1 = (layer_x, layer_y)
                pt2 = (layer_x + layer_w, layer_y + layer_h)
                filled_frame = rectangle(black_bg.copy(), pt1, pt2, 255, -1)
            else:
                # 否则，对掩膜进行腐蚀和膨胀操作，去除小点，以生成填充的框架
                filled_frame = erode(layer_mask, kernel5)
                filled_frame = dilate(filled_frame, kernel5)
            grid_masks.append(filled_frame)

            px_pts = transpose(nonzero(filled_frame))
            px_area = px_pts.shape[0]
            all_x = px_pts[:, 1]
            all_y = px_pts[:, 0]
            all_x.sort()
            all_y.sort()

            grid_x = min(all_x)
            grid_y = min(all_y)
            grid_w = max(all_x) - min(all_x)
            grid_h = max(all_y) - min(all_y)
            grid_br = (grid_x, grid_y, grid_w, grid_h)
            lef, topp = grid_x, grid_y
            # 偏左了，向右移
            lef += 10
            text_pos = (lef, topp)

            filled_frame_erode10 = erode(filled_frame, kernel10)
            filled_frame_erode10_inv = bitwise_not(filled_frame_erode10)
            outline_mask = bitwise_and(filled_frame, filled_frame, mask=filled_frame_erode10_inv)
            filled_outline = cvtColor(outline_mask, COLOR_GRAY2BGRA)
            # 黑色转为透明
            filled_outline[np.all(filled_outline == rgba_black, axis=2)] = rgba_zero
            # 白色转为颜色
            filled_outline[np.all(filled_outline == rgba_white, axis=2)] = color_bgra
            filled_outline = fromarray(cvtColor(filled_outline, COLOR_BGRA2RGBA))
            panel_pil.paste(filled_outline, pos_0, filled_outline)
        panel_draw.text(text_pos, f'{p + 1}', index_color, font=msyh_font60)
    marked_frames = cvtColor(array(panel_pil), COLOR_RGB2BGR)
    write_pic(marked_frames_jpg, marked_frames)
    return grid_masks, marked_frames


def ocr_by_tesseract_simple(image, media_lang, vert=False):
    # 获取所选语言的配置
    language_config = tesseract_language_options[media_lang]
    config = f'-c preserve_interword_spaces=1 --psm 6 -l {language_config}'
    # 如果识别竖排文本
    if vert:
        config += " -c textord_old_baselines=0"
    text = image_to_string(image, config=config)
    return text.strip()


def ocr_by_tesseract(image, media_lang, vert=False):
    # 获取所选语言的配置
    language_config = tesseract_language_options[media_lang]
    config = f'-c preserve_interword_spaces=1 --psm 6 -l {language_config}'
    # 如果识别竖排文本
    if vert:
        config += " -c textord_old_baselines=0"
    data = image_to_data(image, config=config, output_type='dict')
    # 将各个键的值组合成元组
    tess_zipped_data = zip(
        data['level'],
        data['page_num'],
        data['block_num'],
        data['par_num'],
        data['line_num'],
        data['word_num'],
        data['left'],
        data['top'],
        data['width'],
        data['height'],
        data['conf'],
        data['text']
    )
    tess_zipped_data = list(tess_zipped_data)
    return tess_zipped_data


# @logger.catch
def ocr_by_vision(image, media_lang):
    languages = [vision_language_options[media_lang]]  # 设置需要识别的语言

    height, width = image.shape[:2]
    image_data = image.tobytes()
    provider = CGDataProviderCreateWithData(None, image_data, len(image_data), None)
    cg_image = CGImageCreate(width, height, 8, 8 * 3, width * 3, CGColorSpaceCreateDeviceRGB(), 0, provider, None,
                             False, kCGRenderingIntentDefault)

    handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

    request = VNRecognizeTextRequest.new()
    # 使用快速识别（0-快速，1-精确）
    request.setRecognitionLevel_(0)
    # 使用精确识别（0-快速，1-精确）
    # request.setRecognitionLevel_(1)
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
            x_raw = bounding_box.origin.x
            y_raw = bounding_box.origin.y
            w_raw = bounding_box.size.width
            h_raw = bounding_box.size.height
            x = int(x_raw * width)
            y = int(y_raw * height)
            w = int(w_raw * width)
            h = int(h_raw * height)
            # line_br = (x, y, w, h)
            tup = (x, y, w, h, confidence, text)
            text_results.append(tup)
        return text_results
    else:
        print("Error: ", error)
        return []


# @logger.catch
def ocr_by_paddle(image, language):
    lang_code = paddleocr_language_map.get(language, language)
    if lang_code == 'en' and init_ocr:
        ocr = en_ocr
    else:
        ocr = PaddleOCR(use_gpu=False, lang=lang_code)
    result = ocr.ocr(image)
    result0 = result[0]
    text_results = []
    if result0:
        for res in result0:
            para_pts = res[0]
            text, confidence = res[1]
            pts = list(chain(*para_pts))
            pts = [int(x) for x in pts]
            tup = tuple(pts) + (confidence, text)
            text_results.append(tup)
    return text_results


# @logger.catch
def ocr_by_easy(image, language):
    lang_code = easyocr_language_map.get(language, language)
    if lang_code == 'English' and init_ocr:
        reader = en_reader
    else:
        reader = Reader([lang_code])
    result = reader.readtext(image)
    text_results = []
    for res in result:
        # logger.debug(f'{res=}')
        para_pts = res[0]
        text = res[1]
        confidence = res[2]
        pts = list(chain(*para_pts))
        pts = [int(x) for x in pts]
        tup = tuple(pts) + (confidence, text)
        # logger.debug(f'{tup=}')
        text_results.append(tup)
    return text_results


# language_type
# 识别语言类型，默认为CHN_ENG
# 可选值包括：
# - CHN_ENG：中英文混合
# - ENG：英文
# - JAP：日语
# - KOR：韩语
# - FRE：法语
# - SPA：西班牙语
# - POR：葡萄牙语
# - GER：德语
# - ITA：意大利语
# - RUS：俄语

# detect_direction
# 是否检测图像朝向，默认不检测，即：false。朝向是指输入图像是正常方向、逆时针旋转90/180/270度。可选值包括:
# - true：检测朝向；
# - false：不检测朝向。

# detect_language
# 是否检测语言，默认不检测。当前支持（中文、英语、日语、韩语）

# paragraph
# 是否输出段落信息

# probability
# 是否返回识别结果中每一行的置信度
def ocr_by_baidu_base(image, language, req_type):
    _, buffer = imencode('.jpg', image)
    image_as_bytes = buffer.tobytes()

    obd_lang = 'ENG'
    options = {
        # 'detect_direction': 'true',
        'language_type': obd_lang,
        # 'detect_language': 'true',
        'paragraph': 'true',
        'probability': 'true',
    }

    # 调用通用文字识别接口
    if req_type == 'basic':
        result = aip_client.basicGeneral(image_as_bytes, options)
    else:
        result = aip_client.basicAccurate(image_as_bytes, options)

    text_results = []
    paras_result = result['paragraphs_result']
    words_result = result['words_result']
    for p in range(len(paras_result)):
        para = paras_result[p]
        words_result_idx = para['words_result_idx']
        for ind in words_result_idx:
            words_text = words_result[ind]['words']
            probability = words_result[ind]['probability']
            prob_average = probability['average']
            prob_min = probability['min']
            prob_variance = probability['variance']
            tup = (p, ind, prob_average, prob_min, prob_variance, words_text)
            text_results.append(tup)
    return text_results


def ocr_by_baidu(image, language):
    return ocr_by_baidu_base(image, language, req_type='basic')


def ocr_by_baidu_accu(image, language):
    return ocr_by_baidu_base(image, language, req_type='accurate')


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


def get_grouped_bulks(single_cnts_grid, thres=0):
    # 将在Y轴方向上有重叠的部分划分到一起
    grouped_bulks = []
    current_group = [single_cnts_grid[0]]
    for i in range(1, len(single_cnts_grid)):
        curr_cnt = single_cnts_grid[i]
        group_bottom = max([cnt.br_v for cnt in current_group])
        group_top = min([cnt.br_y for cnt in current_group])
        # 如果当前cnt的顶部在组内任何一个cnt的底部之上（考虑阈值）并且
        # 当前cnt的底部在组内任何一个cnt的顶部之下（考虑阈值）
        if curr_cnt.br_y <= group_bottom + thres and curr_cnt.br_v >= group_top - thres:
            current_group.append(curr_cnt)
        else:
            grouped_bulks.append(current_group)
            current_group = [curr_cnt]
    if current_group:
        grouped_bulks.append(current_group)
    return grouped_bulks


# @logger.catch
def sort_bubble(cnt, ax, origin_x, origin_y):
    val = ax * (cnt.cx - origin_x) + (cnt.cy - origin_y)
    if y_first:
        if cnt.core_br_y - origin_y <= 60:
            val = 0.1 * ax * (cnt.cx - origin_x) + (cnt.cy - origin_y)
        elif cnt.core_br_y - origin_y <= 300:
            val = 0.3 * ax * (cnt.cx - origin_x) + (cnt.cy - origin_y)
        elif cnt.core_br_y - origin_y <= 600:
            val = 0.5 * ax * (cnt.cx - origin_x) + (cnt.cy - origin_y)
    return val


# @logger.catch
@timer_decorator
def get_ordered_cnts(single_cnts, image_raw, grid_masks, bubble_order_strs, media_type):
    ih, iw = image_raw.shape[0:2]
    black_bg = zeros((ih, iw), uint8)

    if media_type == 'Manga':
        ax = -1
    else:
        ax = 1
    all_contours = [single_cnt.contour for single_cnt in single_cnts]
    white_bubbles = drawContours(black_bg.copy(), all_contours, -1, 255, FILLED)
    # ================通过画格重新排序气泡框架================
    single_cnts_grids = []
    for g in range(len(grid_masks)):
        grid_mask = grid_masks[g]
        grid_mask_px_pts = transpose(nonzero(grid_mask))
        grid_mask_px_area = grid_mask_px_pts.shape[0]
        all_x = grid_mask_px_pts[:, 1]
        all_y = grid_mask_px_pts[:, 0]
        all_x.sort()
        all_y.sort()
        grid_x = min(all_x)
        grid_y = min(all_y)
        grid_w = max(all_x) - min(all_x)
        grid_h = max(all_y) - min(all_y)
        grid_white_bubbles = bitwise_and(white_bubbles, grid_mask)
        grid_white_bubbles_px_pts = transpose(nonzero(grid_white_bubbles))
        grid_white_bubbles_px_area = grid_white_bubbles_px_pts.shape[0]
        # ================如果画格内有气泡================
        if grid_white_bubbles_px_area >= 300:
            single_cnts_grid = []
            # ================获取当前画格内所有气泡轮廓================
            for s in range(len(single_cnts)):
                single_cnt = single_cnts[s]
                if grid_mask[single_cnt.cy, single_cnt.cx] == 255:
                    single_cnts_grid.append(single_cnt)
            # ================当前画格有可能没有气泡但包含其他画格的气泡的一部分================
            if single_cnts_grid:
                single_cnts_grid = sorted(single_cnts_grid, key=lambda x: x.br_y)
                # ================获取高度块================
                grouped_bulks = get_grouped_bulks(single_cnts_grid)
                # ================如果grid_mask又宽又矮================
                if (grid_mask_px_area <= 0.25 * ih * iw and grid_w >= 0.5 * iw) or fully_framed:
                    # ================所有高度块强制设置为同一组================
                    grouped_bulks = [list(chain(*grouped_bulks))]
                bulk_cnts_group = []
                for g in range(len(grouped_bulks)):
                    grouped_bulk = grouped_bulks[g]
                    y_min = min([cnt.br_y for cnt in grouped_bulk])
                    y_max = max([cnt.br_v for cnt in grouped_bulk])
                    # ================获取团块================
                    mask_y = zeros_like(grid_white_bubbles)
                    mask_y[y_min:y_max, :] = 255
                    masked_grid_white_part = bitwise_and(grid_white_bubbles, mask_y)
                    masked_grid_mask = bitwise_and(grid_mask, mask_y)
                    dilated_masked = dilate(masked_grid_white_part.copy(), kernel60, iterations=1)
                    dilated_masked = bitwise_and(dilated_masked, masked_grid_mask)
                    bulk_contours, _ = findContours(dilated_masked, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
                    bulk_cnts = []
                    for b in range(len(bulk_contours)):
                        bulk_contour = bulk_contours[b]
                        bulk_cnt = Contr(bulk_contour)
                        if bulk_cnt.area >= 300:
                            bulk_cnt.get_core_br(black_bg)
                            bulk_cnts.append(bulk_cnt)
                    all_br_x = [cnt.core_br_x for cnt in bulk_cnts]
                    all_br_y = [cnt.core_br_y for cnt in bulk_cnts]
                    origin_x, origin_y = min(all_br_x), min(all_br_y)
                    bulk_cnts.sort(key=lambda x: ax * (x.cx - origin_x) + (x.cy - origin_y))
                    bulk_cnts_group.append(bulk_cnts)
                bulk_cnts_grid = list(chain(*bulk_cnts_group))
            if bulk_cnts_grid:
                cnts_in_bulk_grid = []
                for b in range(len(bulk_cnts_grid)):
                    bulk_cnt = bulk_cnts_grid[b]
                    cnts_in_bulk = [x for x in single_cnts if
                                    pointPolygonTest(bulk_cnt.contour, (x.cx, x.cy), False) > 0]
                    if cnts_in_bulk:
                        # ================获取整体的外接矩形左上角坐标================
                        for c in range(len(cnts_in_bulk)):
                            cnt = cnts_in_bulk[c]
                            cnt.get_core_br(black_bg)
                        all_br_x = [cnt.core_br_x for cnt in cnts_in_bulk]
                        all_br_y = [cnt.core_br_y for cnt in cnts_in_bulk]
                        origin_x, origin_y = min(all_br_x), min(all_br_y)
                        # ================画格内按坐标排序================
                        cnts_in_bulk.sort(key=lambda cnt: sort_bubble(cnt, ax, origin_x, origin_y))
                        cnts_in_bulk_grid.append(cnts_in_bulk)
                single_cnts_grid_ordered = list(chain(*cnts_in_bulk_grid))
                if bubble_order_strs:
                    # 根据bubble_order_strs中的cxy_str顺序对single_cnts_grid_ordered进行排序
                    single_cnts_grid_ordered.sort(key=lambda cnt: bubble_order_strs.index(cnt.cxy_str))
                single_cnts_grids.append(single_cnts_grid_ordered)
    return single_cnts_grids


# @logger.catch
def order1pic(img_file, frame_data, order_data, all_masks, media_type):
    logger.warning(f'{img_file=}')
    img_folder = img_file.parent
    auto_subdir = Auto / img_folder.name
    order_preview_jpg = auto_subdir / f'{img_file.stem}-气泡排序.jpg'
    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    ih, iw = image_raw.shape[0:2]
    # ================矩形画格信息================
    # 从 frame_data 中获取画格信息，默认为整个图像的矩形区域
    frame_grid_strs = frame_data.get(img_file.name, [f'0,0,{iw},{ih}~0,0,{iw},{ih}'])
    bubble_order_strs = order_data.get(img_file.name, [])
    grid_masks, marked_frames = get_grid_masks(img_file, frame_grid_strs)
    # ================获取对应的文字图片================
    mask_pics = [x for x in all_masks if x.stem.startswith(img_file.stem)]
    ordered_cnts = []
    if mask_pics:
        single_cnts = get_single_cnts(image_raw, mask_pics)
        logger.debug(f'{len(single_cnts)=}')
        single_cnts_grids = get_ordered_cnts(single_cnts, image_raw, grid_masks, bubble_order_strs, media_type)
        ordered_cnts = list(chain(*single_cnts_grids))
        order_preview = get_order_preview(marked_frames, single_cnts_grids)
        write_pic(order_preview_jpg, order_preview)
    return img_file, ordered_cnts


@timer_decorator
def get_order_preview(marked_frames, single_cnts_grids):
    ordered_cnts = list(chain(*single_cnts_grids))
    image_pil = fromarray(cvtColor(marked_frames, COLOR_BGR2RGB))
    # 创建与原图大小相同的透明图像
    bubble_overlay = Image.new('RGBA', image_pil.size, rgba_zero)
    core_br_overlay = Image.new('RGBA', image_pil.size, rgba_zero)
    line_overlay = Image.new('RGBA', image_pil.size, rgba_zero)
    bubble_draw = ImageDraw.Draw(bubble_overlay)
    core_br_draw = ImageDraw.Draw(core_br_overlay)
    line_draw = ImageDraw.Draw(line_overlay)
    line_alpha = 192
    # ================绘制示意图================
    for f in range(len(ordered_cnts)):
        bubble_cnt = ordered_cnts[f]
        contour_points = [tuple(point[0]) for point in bubble_cnt.contour]
        x, y, w, h = bubble_cnt.core_br
        # 从 tab20 颜色映射中选择一个颜色
        color = colormap_tab20(f % 20)[:3]
        bubble_color_rgba = tuple(int(c * 255) for c in color) + (int(255 * bubble_alpha),)
        bubble_draw.polygon(contour_points, fill=bubble_color_rgba)
        color_rgba = tuple(int(c * 255) for c in color) + (255,)
        # 绘制矩形
        core_br_draw.rectangle([(x, y), (x + w, y + h)], outline=color_rgba, width=2)

    for f in range(len(ordered_cnts)):
        bubble_cnt = ordered_cnts[f]
        # 在轮廓中心位置添加序数
        text = str(f + 1)
        text_bbox = bubble_draw.textbbox(bubble_cnt.cxy, text, font=msyh_font100, anchor="mm")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        xy_pos = (bubble_cnt.cx - text_width // 2, bubble_cnt.cy - text_height // 2)
        bubble_draw.text(xy_pos, text, font=msyh_font100, fill=semi_transparent_purple)

    # ================按照画格内对话框顺序添加红色连线================
    for g in range(len(single_cnts_grids)):
        single_cnts_grid = single_cnts_grids[g]
        # 确定该画格内需要绘制的红线数量
        total_lines = len(single_cnts_grid) - 1
        if len(single_cnts_grid) >= 2:
            for b in range(len(single_cnts_grid)):
                bubble_cnt = single_cnts_grid[b]
                # 绿色圆圈表示质心
                line_draw.ellipse([(bubble_cnt.cxy[0] - 10, bubble_cnt.cxy[1] - 10),
                                   (bubble_cnt.cxy[0] + 10, bubble_cnt.cxy[1] + 10)], fill=(0, 255, 0, line_alpha))
                # 如果不是该画格内的第一个气泡，与上一个气泡的质心画连线
                if b > 0:
                    prev_cnt = single_cnts_grid[b - 1]
                    line_color = (255, 0, 0, line_alpha)
                    if total_lines >= 2:
                        # 根据当前红线的序号逐渐增加
                        blue_value = int(255 * (b - 1) / total_lines)
                        # 使用计算出的蓝色值来调整红线的颜色
                        line_color = (255, 0, blue_value, line_alpha)
                    # 在新的透明图层上用调整后的颜色绘制连线
                    line_draw.line([prev_cnt.cxy, bubble_cnt.cxy], fill=line_color, width=5)

    # 将带有半透明颜色轮廓的透明图像与原图混合
    image_pil.paste(bubble_overlay, mask=bubble_overlay)
    image_pil.paste(core_br_overlay, mask=core_br_overlay)
    image_pil.paste(line_overlay, mask=line_overlay)

    # 将 PIL 图像转换回 OpenCV 图像
    blended_img = cvtColor(array(image_pil), COLOR_RGB2BGR)
    return blended_img


def get_line_imgs(cropped_img, c):
    gray_cropped_img = cvtColor(cropped_img, COLOR_BGR2GRAY)
    # 二值化图像
    ret, mask_img = threshold(gray_cropped_img, 127, 255, THRESH_BINARY)
    # 创建水平投影图像
    hori_proj_pil, hori_proj_data = get_proj_img(mask_img, 'horizontal')
    # 创建垂直投影图像
    ver_proj_pil, ver_proj_data = get_proj_img(mask_img, 'vertical')

    if do_dev_pic:
        cropped_img_png = current_dir / f'cropped_img_{c}.png'
        mask_img_png = current_dir / f'mask_img_{c}.png'
        hori_proj_img_png = current_dir / f'hori_proj_pil_{c}.png'
        ver_proj_img_png = current_dir / f'ver_proj_pil_{c}.png'
        write_pic(cropped_img_png, cropped_img)
        write_pic(mask_img_png, mask_img)
        write_pic(hori_proj_img_png, hori_proj_pil)
        write_pic(ver_proj_img_png, ver_proj_pil)

    hori_proj_black = array(hori_proj_pil)
    hori_proj_white = bitwise_not(hori_proj_black)

    projection_contours, hierarchy = findContours(hori_proj_white, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    text_projection_cnts = []
    for p in range(len(projection_contours)):
        contour = projection_contours[p]
        cnt = Contr(contour)
        # ================对轮廓数值进行初筛================
        condition_p1 = 100 <= cnt.area <= 1000000
        condition_p2 = 80 <= cnt.perimeter <= 50000
        condition_ps = [
            condition_p1,
            condition_p2,
        ]
        # logger.debug(f'{condition_ps=}, {cnt.area=:.1f}, {cnt.perimeter=:.2f}')
        if all(condition_ps):
            text_projection_cnts.append(cnt)
    text_projection_cnts.sort(key=lambda x: x.br_y)

    line_imgs = []
    for t in range(len(text_projection_cnts)):
        text_projection_cnt = text_projection_cnts[t]
        # 获取轮廓的上下边界来提取行文字图像
        line_img = cropped_img[text_projection_cnt.br_y - 1:text_projection_cnt.br_v + 1, :]
        line_imgs.append(line_img)
    return line_imgs


def tesseract2text(tess_zipped_data):
    tess_zipped_data5 = [x for x in tess_zipped_data if x[0] == 5]
    par_nums = [x[3] for x in tess_zipped_data5]
    par_nums = reduce_list(par_nums)
    par_nums.sort()
    line_nums = [x[4] for x in tess_zipped_data5]
    line_nums = reduce_list(line_nums)
    line_nums.sort()
    lines_tesseract = []
    for par_num in par_nums:
        for line_num in line_nums:
            line_data = [x for x in tess_zipped_data5 if x[3] == par_num and x[4] == line_num]
            line_data.sort(key=lambda x: x[5])
            for l in range(len(line_data)):
                word_data = line_data[l]
                level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text = word_data
            line_words = [x[-1] for x in line_data]
            line_text = ' '.join(line_words)
            if line_text != '':
                lines_tesseract.append(line_text)
    return lines_tesseract


# 'level': 层次结构中的级别。1 表示页面级别，2 表示区域级别，3 表示段落级别，4 表示文本行级别，5 表示单词级别。
# 'page_num': 页码。
# 'block_num': 区域编号。
# 'par_num': 段落编号。
# 'line_num': 行号。
# 'word_num': 单词编号。
# 'left': 边界框左上角的 x 坐标。
# 'top': 边界框左上角的 y 坐标。
# 'width': 边界框的宽度。
# 'height': 边界框的高度。
# 'conf': 置信度，表示 Tesseract 识别结果的可信度，范围为 0 到 100。
# 'text': 识别的文本。
def tesseract2meta(img_np, tess_zipped_data, ocr_para):
    # 从docx段落中提取带格式的单词
    ocr_lines = []
    current_line = []
    has_bold = False
    has_italic = False

    for run in ocr_para.runs:
        run_text_parts = run.text.split('\n')
        for i, part in enumerate(run_text_parts):
            for word in part.split():
                word_format = []
                if run.bold:
                    word_format.append("b")
                    has_bold = True
                if run.italic:
                    word_format.append("i")
                    has_italic = True
                current_line.append((word, ''.join(word_format)))
            # 如果当前部分不是最后一个部分，说明有一个换行符，需要开始新的行
            if i < len(run_text_parts) - 1:
                ocr_lines.append(current_line)
                current_line = []

    # 添加最后一行
    if current_line:
        ocr_lines.append(current_line)

    ocr_lines = ocr_lines[1:]
    # 段落的格式描述
    if has_bold and has_italic:
        para_format = "bi"
    elif has_bold:
        para_format = "b"
    elif has_italic:
        para_format = "i"
    else:
        para_format = ""

    # 获取气泡元数据
    bubble_meta_str = ocr_para.runs[0].text
    bubble_meta_list = bubble_meta_str.split('~')
    coors_str = bubble_meta_list[0]
    src_font_size = bubble_meta_list[1].removeprefix('S')
    dst_font_size = bubble_meta_list[2].removeprefix('D')
    src_font_size = int(src_font_size)
    dst_font_size = int(dst_font_size)
    bubble_shape = bubble_meta_list[3]
    text_direction = bubble_meta_list[4]
    text_alignment = bubble_meta_list[5]
    bubble_color_str = ''
    if len(bubble_meta_list) == 9:
        bubble_color_str = bubble_meta_list[-3]
    letter_color_str = bubble_meta_list[-2]
    letter_font_name = bubble_meta_list[-1]
    # 获取气泡坐标
    coors = coors_str.split(',')
    coors = [int(x) for x in coors]
    min_x, min_y, br_w, br_h = coors

    # 计算中心点坐标
    center_x = min_x + br_w / 2
    center_y = min_y + br_h / 2
    center_pt = (int(center_x), int(center_y))

    tess_zipped_data5 = [x for x in tess_zipped_data if x[0] == 5]
    par_nums = [x[3] for x in tess_zipped_data5]
    par_nums = reduce_list(par_nums)
    par_nums.sort()
    line_nums = [x[4] for x in tess_zipped_data5]
    line_nums = reduce_list(line_nums)
    line_nums.sort()
    words_w_format = []
    line_infos = []
    for par_num in par_nums:
        # 每个段落
        for line_num in line_nums:
            # 每一行
            line_data = [x for x in tess_zipped_data5 if x[3] == par_num and x[4] == line_num]
            line_data.sort(key=lambda x: x[5])
            word_imgs = []
            for l in range(len(line_data)):
                # 每个单词
                word_data = line_data[l]
                level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text = word_data
                word_br = (left, top, width, height)
                br_area = width * height
                # 裁剪出单词
                crop_br = (top, top + height, left, left + width)
                word_img = img_np[top - 1:top + height + 1, left - 1:left + width + 1]
                # 检查图像是否为空或维度是否为0
                if word_img.size == 0 or word_img.shape[0] == 0 or word_img.shape[1] == 0:
                    logger.error(f'{crop_br=}')
                elif text != '':
                    pos_meta = (text, par_num, line_num, word_num, left, top, width, height, conf, word_img)
                    word_imgs.append(pos_meta)
            if word_imgs:
                line_infos.append(word_imgs)

    if len(ocr_lines) == len(line_infos):
        # 行数相等时
        for l in range(len(line_infos)):
            # 每一行
            word_imgs = line_infos[l]
            ocr_line = ocr_lines[l]
            tess_words = [x[0] for x in word_imgs]
            ocr_words = [x[0] for x in ocr_line]
            if len(word_imgs) == len(ocr_line):
                # 词数相等时
                for w in range(len(word_imgs)):
                    # 每一词
                    pos_meta = word_imgs[w]
                    text, par_num, line_num, word_num, left, top, width, height, conf, word_img = pos_meta
                    word_img_png = current_dir / f'word_{par_num}_{line_num}_{word_num}.png'
                    write_pic(word_img_png, word_img)
                    # 计算裁剪出的单词图片的黑色像素面积
                    # 检查是否为 NumPy 数组，如果是，将其转换为 PIL 图像
                    if isinstance(word_img, ndarray):
                        word_img = fromarray(word_img)
                    # 转换为灰度图像
                    gray_img = word_img.convert('L')
                    # 转换为二值图像
                    binary_img = gray_img.point(lambda x: 0 if x <= bit_thres else 255, '1')
                    # 计算黑色像素的数量（假设黑色像素的值为0）
                    black_px_area = binary_img.histogram()[0]
                    text_w_format = ocr_line[w]
                    format_tuple = text_w_format + (height, black_px_area,)
                    words_w_format.append(format_tuple)
            else:
                logger.error(f'词数不相等, {tess_words=}, {ocr_words=}')
    else:
        logger.error(f'行数不相等,{ocr_para.text=}')
    return bubble_color_str, letter_color_str, words_w_format


def get_ocr_data(ocr_engine, pic_ocr_data, img_np, media_lang, elements_num):
    if ocr_engine in pic_ocr_data:
        ocr_results = form2data(pic_ocr_data[ocr_engine])
    else:
        ocr_results = globals()[f'ocr_by_{ocr_engine.lower()}'](img_np, media_lang)
        ocr_results_form = []
        for row in ocr_results:
            row_nums = row[:-elements_num]
            row_str = ','.join(map(str, row_nums))
            text = row[-1]
            confs = row[-elements_num:-1]
            if confs:
                row_str += ','
                row_str += ','.join([f'{x:.2f}' for x in confs])
            row_str += f'|{text}'
            ocr_results_form.append(row_str)
        pic_ocr_data[ocr_engine] = ocr_results_form
    return ocr_results, pic_ocr_data


def rec2text(rec_results):
    lines_vision = []
    current_line = []
    current_y = None
    for r in range(len(rec_results)):
        rec_result = rec_results[r]
        y = rec_result[1]
        text_seg = rec_result[-1]
        good_text = "".join([similar_chars_map.get(c, c) for c in text_seg])
        # 检查 y 值
        if current_y is not None and abs(y - current_y) <= y_thres:
            # 如果 y 值相近，添加到当前行
            current_line.append(good_text)
        else:
            # 否则，开始新的一行
            if current_line:
                lines_vision.append(' '.join(current_line))
            current_line = [good_text]
            current_y = y

    # 添加最后一行
    if current_line:
        lines_vision.append(' '.join(current_line))
    return lines_vision


def get_upper_ratio(input_text):
    # 计算文本中大写和小写字母的数量
    letters = [char for char in input_text if char.isalpha()]
    letter_count = Counter([char.isupper() for char in letters])
    # 计算大写字母的百分比
    up_ratio = 0
    if letters:
        up_ratio = letter_count[True] / len(letters)
    return up_ratio


def correct_word(word_text):
    for old, new in corrections:
        if old in word_text:
            rep_word = word_text.replace(old, new)
            if rep_word.lower() in good_words:
                return rep_word

    if word_text.endswith('IS'):
        new_word_text = word_text.removesuffix('IS')
        if new_word_text.lower() in good_words or new_word_text.capitalize() in good_words:
            return f"{new_word_text}'S"

    if word_text.endswith(('Z', '2')):
        new_word_text = word_text.removesuffix('Z').removesuffix('2')
        if new_word_text.lower() in good_words or new_word_text.capitalize() in good_words:
            return f"{new_word_text}?"

    return word_text


# @logger.catch
def better_text(input_text, ocr_type):
    text_format = input_text.strip()
    # 根据替换规则表进行替换
    for old, new in replace_rules.items():
        text_format = text_format.replace(old, new)
    # ================处理省略号================
    text_format = sub(r'(\. ?){2,}', '…', text_format)
    text_format = sub(r'-{3,}', '--', text_format)
    # ================处理引号================
    for old, new in replacements.items():
        text_format = text_format.replace(old, new)
    # ================处理0================
    text_format = sub(r'0{3,}', lambda x: 'O' * len(x.group()), text_format)
    # ================处理有缩写符号的词================
    for old, new in better_abbrs.items():
        if old == "15":
            text_format = re.sub(r'\b15\b(?!\s*percent\b)', new, text_format, flags=IGNORECASE)
        else:
            text_format = re.sub(r'\b' + old + r'\b', new, text_format)

    letters = [char for char in input_text if char.isalpha()]
    up_ratio = get_upper_ratio(text_format)
    check_upper = False
    if up_ratio > 0.5 and len(letters) >= 6:
        check_upper = True

    if ocr_type in ['tesseract', 'vision', 'baidu', 'baidu_accu']:
        lines = text_format.splitlines()
        processed_lines = []
        for l in range(len(lines)):
            line = lines[l]
            words = list(finditer(r'\b\w+\b', line))
            new_words = []
            for w in range(len(words)):
                word = words[w]
                start, end = word.span()
                word_text = line[start:end]
                if not word_text.lower() in good_words:
                    word_text = correct_word(word_text)
                # 处理大小写
                if word_text.isupper() or word_text.lower().startswith(interjections):
                    new_word = word_text
                elif word_text[0].isdigit():
                    new_word = word_text
                elif check_upper:
                    new_word = word_text.upper()
                else:
                    new_word = word_text
                new_words.append(new_word)
            if new_words:
                line = sub(r'\b\w+\b', '{}', line).format(*new_words)
            else:
                line = sub(r'\b\w+\b', '', line)
            processed_lines.append(line)
        text_format = lf.join(processed_lines)

    # ================处理标点格式================
    # 确保"…"前后没有空格或特殊字符
    text_format = sub(r' +… +| +…|… +', '…', text_format)
    text_format = sub(r'-+…|…-+|\*+…|…\*+', '…', text_format)
    # 确保"-"前后没有空格
    text_format = sub(r'- +| +-', '-', text_format)
    # 其他替换
    text_format = sub(r' *• *', ' ', text_format)
    text_format = sub(r' +([?!.,])', r'\1', text_format)
    # 在标点后的单词前添加空格
    text_format = sub(r'([?!.,])(?![0-9])(\w)', r'\1 \2', text_format)
    text_format = text_format.replace('#$0%', '#$@%')
    text_format = text_format.replace('D. C.', 'D.C.')
    return text_format


def find_punct(text):
    """找到文本的段首和段尾标点"""
    start_punct = match(r'^\W+', text)
    end_punct = search(r'\W+$', text)
    start_p = start_punct.group() if start_punct else ''
    end_p = end_punct.group() if end_punct else ''
    if start_p == '. ':
        start_p = '…'
    return start_p, end_p


def better_apostrophe(text):
    """
    优化文本中的撇号（apostrophe）使用。
    """
    # 将文本合并为单行
    merged_text = ' '.join(text.splitlines())

    # 使用Spacy进行自然语言处理
    doc = nlp(merged_text)

    # 记录需要插入撇号的位置
    insert_positions = []

    for i, token in enumerate(doc):
        ori_text = token.text  # 原始词文本
        lower_text = ori_text.lower()  # 转换为小写
        next_token = doc[i + 1] if i + 1 < len(doc) else None  # 获取下一个词（如果存在）

        if lower_text == "worlds":
            # 处理 "worlds" 和 "world's"
            has_noun_child = any(child.pos_ == "NOUN" for child in token.children)  # 检查是否有名词子节点
            if has_noun_child or (next_token and (next_token.pos_ == "VERB" or next_token.pos_ == "NOUN")):
                matches = [m.end() for m in finditer(r'\b' + escape(ori_text) + r'\b', text)]
                if matches:
                    closest_match = min(matches, key=lambda x: abs(x - token.idx))
                    insert_positions.append(closest_match - 1)
        elif lower_text == "its":
            # 处理 "its" 和 "it's"
            if next_token and (next_token.pos_ == "VERB" and next_token.lemma_ not in ["be", "have"]):
                matches = [m.end() for m in finditer(r'\b' + escape(ori_text) + r'\b', text)]
                if matches:
                    closest_match = min(matches, key=lambda x: abs(x - token.idx))
                    insert_positions.append(closest_match - 1)

    # 根据之前的判断结果，在正确的位置插入撇号
    for pos in reversed(insert_positions):  # 从后往前插入，以避免影响位置
        text = text[:pos] + "'" + text[pos:]
    return text


def is_valid_tense(word):
    doc = nlp(word)
    for token in doc:
        if token.tag_ in ['VBD', 'VBG']:  # VBD: past tense, VBG: gerund/present participle
            return True
    return False


def calculate_similarity(lines1, lines2):
    total_similarity = 0
    for line1, line2 in zip(lines1, lines2):
        s = SequenceMatcher(None, line1, line2)
        total_similarity += s.ratio()
    return total_similarity / len(lines1) if lines1 else 0


def get_final_token(token1, token2, up_ratio, is_sent_start, i1, i2, text1, line1):
    # 找到这个字母所在的完整单词
    word_start, word_end = i1, i2
    while word_start > 0 and line1[word_start - 1].isalpha():
        word_start -= 1
    while word_end < len(line1) and line1[word_end].isalpha():
        word_end += 1
    ori_word = line1[word_start:word_end]
    rep_word = ori_word[:i1 - word_start] + token2 + ori_word[i2 - word_start:]
    ori_word_lower = ori_word.lower()
    rep_word_lower = rep_word.lower()
    final_token = token1
    # 检查是否是单个字母替换成单个字母的情况
    if len(token1) == 1 and len(token2) == 1 and token1.lower() == token2.lower():
        if up_ratio >= 0.9 or is_sent_start or ori_word in proper_nouns:
            # 全大写或句首
            final_token = token1.upper()
        else:
            final_token = token1.lower()
        if final_token != token1:
            print(f'{token1}->{final_token}')
    elif token1 in punct_rep_rules and token2 in punct_rep_rules[token1]:
        final_token = token2
    elif token1.isdigit() and len(token1) == 1 and all(ch in valid_chars for ch in token2):
        final_token = token2
    elif token1.isdigit() and len(token1) == 1 and token2.isdigit() and len(token2) == 1:
        final_token = token2
    elif token1 == 'I.' and token2 == 'T…':
        final_token = 'I…'
    elif len(token1) == 1 and len(token2) == 1 and token1.isalpha():
        # 尝试获取单数形式
        ori_word_singular = lemmatizer.lemmatize(ori_word_lower, pos='n')
        rep_word_singular = lemmatizer.lemmatize(rep_word_lower, pos='n')
        # 检查这个单词是否是一个“合理的”单词
        if ori_word_lower in interjections:
            # 如果是语气词，保留原词
            final_token = token1
        elif ori_word_lower in good_words or ori_word_singular in good_words or ori_word_lower in fine_words:
            final_token = token1
        else:
            if token2.isalpha():
                if ori_word_lower.endswith(('ing', 'ed')):
                    if is_valid_tense(ori_word):
                        # 如果是过去时或进行时的合理词，保留原词
                        final_token = token1
                    elif rep_word_lower in good_words or rep_word_lower in fine_words:
                        print(f'{ori_word}->{rep_word}')
                        # 否则，如果替换后的词是合理的，进行替换
                        final_token = token2
                    else:
                        # 否则，保留原词
                        final_token = token1
                elif rep_word_lower in good_words or rep_word_singular in good_words or rep_word_lower in fine_words:
                    final_token = token2
                    print(f'{ori_word}->{rep_word}')
                else:
                    final_token = token1
            else:
                final_token = token2
                print(f'{ori_word}->{rep_word}')
    elif token1 == ',' and token2 == '.':
        # 获取原始文本和添加标点后的文本中的句子
        adj_text1 = text1[:i1] + token2 + text1[i2:]
        final_token = adjust_sent(token1, token2, text1, adj_text1)
    else:
        final_token = token1
    return final_token


def get_sents(text):
    """获取文本中的所有句子的起始和结束位置"""
    # 考虑句号、问号、感叹号、省略号等作为句子的结束
    sentence_boundaries = finditer(r'(?<=[.!?…])\s+', text)
    start = 0
    sentences = []
    for match in sentence_boundaries:
        end = match.start()
        sentences.append((start, end))
        start = match.end()
    sentences.append((start, len(text)))
    return sentences


def adjust_sent(token1, token2, text1, adj_text1):
    final_token = token1
    ori_sents = get_sents(text1)
    mod_sents = get_sents(adj_text1)

    # 比较两个版本的句子结构，找到句子二和句子三的位置
    split_index = None
    for idx, (orig_sent, mod_sent) in enumerate(zip(ori_sents, mod_sents)):
        if orig_sent != mod_sent:
            split_index = idx
            break

    # 如果找到了被分割的句子
    if split_index is not None and split_index + 1 < len(mod_sents):
        # 获取句子三的起始和结束位置
        next_sent_i, next_sent_j = mod_sents[split_index + 1]
        next_sent = adj_text1[next_sent_i:next_sent_j]
        print(f'{next_sent=}')
        # 检查句子是否首字母大写且不是整句话全部大写
        if next_sent and next_sent[0].isupper() and not next_sent.isupper():
            final_token = token2
    return final_token


def fix_w_tess(text1, text2):
    # 将输入文本按行分割
    lines1 = text1.strip().splitlines()
    lines2 = text2.strip().splitlines()
    up_ratio = get_upper_ratio(text1)
    text1_merged = ' '.join(lines1)
    doc = nlp(text1_merged)
    sent_starts = [sent.start_char for sent in doc.sents]
    new_lines1 = []

    # 如果 lines1 比 lines2 少一行
    if len(lines1) + 1 == len(lines2):
        # 情况1：少的是第一行
        sim1 = calculate_similarity(lines1, lines2[1:])
        # 情况2：少的是最后一行
        sim2 = calculate_similarity(lines1, lines2[:-1])
        # 选择相似度最高的情况
        if sim1 > sim2:
            lines1 = [None] + lines1
        else:
            lines1 = lines1 + [None]

    # 当前处理到的字符位置
    pin = 0
    for line1, line2 in zip_longest(lines1, lines2, fillvalue=None):
        # 遍历两个文本的每一行
        # 如果 line2 为 None，保持 line1 不变
        if line2 is None:
            new_lines1.append(line1)
            print(f'{line1=}')
        elif line1 is None:
            new_lines1.append(line2)
            print(f'{line2=}')
        else:
            s = SequenceMatcher(None, line1, line2)
            # logger.info(f"{line1},{line2}")
            new_line1 = ''
            # 获取操作码并遍历，以找出两行文本之间的差异
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                token1 = line1[i1:i2]
                token2 = line2[j1:j2]
                last_sent_i = max([x for x in sent_starts if x <= pin], default=0)
                is_sent_start = all(not c.isalpha() for c in text1_merged[last_sent_i:pin])
                # 默认情况下，final_token 是 token1
                final_token = token1

                if tag == 'equal':
                    final_token = token1
                elif tag == 'replace':
                    final_token = get_final_token(token1, token2, up_ratio, is_sent_start, i1, i2, text1, line1)
                elif tag == 'insert':
                    # 使用正则表达式提取行中的所有单词
                    words = findall(r"[\w']+", line1)
                    # 使用finditer()获取每个单词的位置
                    word_positions = [(m.start(), m.end()) for m in finditer(r"[\w']+", line1)]
                    # 根据i1的位置找到正确的单词及其位置
                    ori_word = None
                    word_start = None
                    for word, (start, end) in zip(words, word_positions):
                        if start <= i1 < end:
                            ori_word = word
                            word_start = start
                            break
                    if ori_word is not None and word_start is not None:
                        rep_word = ori_word[:i1 - word_start] + token2 + ori_word[i1 - word_start:]
                    else:
                        word_start = i1
                        word_end = i1
                        while word_start > 0 and line1[word_start - 1].isalpha():
                            word_start -= 1
                        while word_end < len(line1) and line1[word_end].isalpha():
                            word_end += 1
                        ori_word = line1[word_start:word_end]
                        rep_word = ori_word[:i1 - word_start] + token2 + ori_word[i1 - word_start:]
                    if ori_word == "C'ON" and token2 == 'M':
                        # 检查是否是C'ON变成C'MON的情况
                        final_token = token2
                        print(f'[{final_token}]')
                    elif token2 == '-' and i1 > 0 and line1[i1 - 1] == '-':
                        # 如果在-之后插入一个-，接受这个插入
                        final_token = token2
                        print(f'[{final_token}]')
                    elif len(token2) == 1 and token2.isalpha() and len(ori_word) >= 1:
                        # 检查这个单词是否是一个“合理的”单词
                        if ori_word.lower() not in good_words and rep_word.lower() in good_words or rep_word.lower() in fine_words:
                            final_token = token2
                            print(f'[{final_token}]')
                    elif match(r'[a-zA-Z]', token2) or match(r'\s*\d+$', token2) or match(r'\s+', token2):
                        # 忽略纯字母、纯数字和纯空格的插入
                        pass
                    elif token2 in (':', '¥', '?', '! ', '[', ']', '(', ')', '-', '"', 'O'):
                        pass
                    elif token2 in ['.', '. ']:
                        # 获取原始文本和添加标点后的文本中的句子
                        adj_text1 = text1[:i1] + token2 + text1[i1:]
                        final_token = adjust_sent(token1, token2, text1, adj_text1)
                    else:
                        final_token = token2
                        print(f'[{final_token}]')
                elif tag == 'delete':
                    final_token = token1

                pin += len(token1)
                new_line1 += final_token
            # 将处理后的新行添加到结果列表中
            new_lines1.append(new_line1)
            # 计算空格
            pin += 1
    # 将所有新行连接成一个字符串并返回
    new_text1 = lf.join(new_lines1)
    if new_text1 != text1:
        # print(f'{text1=}')
        # print(f'{new_text1=}')
        pass
    return new_text1


def better_punct(vision_text_format, refer_ocrs):
    """优化 vision_text_format 文本"""

    # 计算文本中的单词数量
    word_count = len(list(finditer(r'\b\w+\b', vision_text_format)))

    start_puncts = []
    end_puncts = []

    # 收集所有参考文本的段首和段尾标点
    for text in refer_ocrs:
        start, end = find_punct(text)
        if start:
            start_puncts.append(start)
        if end:
            end_puncts.append(end)

    # 计算出现次数
    start_counter = Counter(start_puncts)
    end_counter = Counter(end_puncts)

    # 找到出现至少两次的段首和段尾标点
    common_start = [k for k, v in start_counter.items() if v >= 2]
    common_end = [k for k, v in end_counter.items() if v >= 2]

    # 获取 vision_text_format 的段首和段尾标点
    vision_start, vision_end = find_punct(vision_text_format)

    # 如果 vision 的段首标点不在共同段首标点中，则替换
    if vision_start not in common_start and common_start:
        vision_text_format = common_start[0] + vision_text_format[len(vision_start):]

    certains = ['to be continued']
    # 如果 vision 的段尾标点不在共同段尾标点中，则替换
    if vision_end:
        if vision_end not in common_end and common_end:
            vision_text_format = vision_text_format[:-len(vision_end)] + common_end[0]
    else:
        if common_end:
            vision_text_format += common_end[0]
        else:
            # 如果没有共同的段尾标点，添加最长的段尾标点
            longest_end_punct = max(end_puncts, key=len, default='')
            if longest_end_punct:
                vision_text_format += longest_end_punct
            elif vision_text_format.lower().endswith(interjections):
                # 语气词
                pass
            elif vision_text_format.lower() in certains:
                # 未完待续等特殊段落
                pass
            elif word_count >= 8:
                # 如果单词数量大于或等于8，添加句号
                vision_text_format += '.'

    if end_counter['-'] >= 1 and end_counter['--'] >= 1:
        vision_text_format = vision_text_format.rstrip('-') + '--'

    # 如果文本以逗号结尾，替换为句号
    vision_text_format = sub(r',$', '.', vision_text_format)
    vision_text_format = vision_text_format.replace('…,', '…')
    tesseract_text_format = refer_ocrs[-1]
    if '¿' in vision_text_format:
        vision_text_format = tesseract_text_format
    vision_text_format = sub(r'\'OR', 'OR', vision_text_format)
    return vision_text_format


def run2ansi(run):
    text = run.text
    if text.strip() == '':
        return text
    elif run.bold and run.italic:
        return f"\033[1;3;35m{text}\033[0m"  # 粗斜体紫色
    elif run.bold:
        return f"\033[1;34m{text}\033[0m"  # 粗体蓝色
    elif run.italic:
        return f"\033[3;32m{text}\033[0m"  # 斜体绿色
    else:
        return text


# @logger.catch
def ocr1pic(img_file, frame_data, order_data, ocr_data, all_masks, media_type, media_lang, vert):
    pic_results = []
    whiten_png = img_file.parent / f'{img_file.stem}-Whiten.png'
    logger.warning(f'{img_file=}')

    stem_tup = ('stem', img_file.stem)
    pic_results.append(stem_tup)

    image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
    ih, iw = image_raw.shape[0:2]
    # ================矩形画格信息================
    # 从 frame_data 中获取画格信息，默认为整个图像的矩形区域
    frame_grid_strs = frame_data.get(img_file.name, [f'0,0,{iw},{ih}~0,0,{iw},{ih}'])
    bubble_order_strs = order_data.get(img_file.name, [])
    grid_masks, marked_frames = get_grid_masks(img_file, frame_grid_strs)

    # ================获取对应的文字图片================
    mask_pics = [x for x in all_masks if x.stem.startswith(img_file.stem)]
    if mask_pics:
        single_cnts = get_single_cnts(image_raw, mask_pics)
        logger.debug(f'{len(single_cnts)=}')

        single_cnts_grids = get_ordered_cnts(single_cnts, image_raw, grid_masks, bubble_order_strs, media_type)
        ordered_cnts = list(chain(*single_cnts_grids))

        for c in range(len(ordered_cnts)):
            single_cnt = ordered_cnts[c]
            color_pattern = single_cnt.color_pattern
            img_np = single_cnt.cropped_img
            img_md5 = generate_md5(img_np)
            cp_bubble, cp_letter = color_pattern
            pic_locate = f'{img_file.stem}, {c}, {img_md5}'
            pic_ocr_data = ocr_data.get(pic_locate, {})

            # ================对裁剪后的图像进行文字识别================
            tess_zipped_data, pic_ocr_data = get_ocr_data('tesseract', pic_ocr_data, img_np, media_lang, 1)
            rec_results_vision, pic_ocr_data = get_ocr_data('vision', pic_ocr_data, img_np, media_lang, 2)
            if 'Paddle' in custom_ocr_engines:
                rec_results_paddle, pic_ocr_data = get_ocr_data('paddle', pic_ocr_data, img_np, media_lang, 2)
            if 'Easy' in custom_ocr_engines:
                rec_results_easy, pic_ocr_data = get_ocr_data('easy', pic_ocr_data, img_np, media_lang, 2)
            if 'Baidu' in custom_ocr_engines:
                rec_results_baidu, pic_ocr_data = get_ocr_data('baidu', pic_ocr_data, img_np, media_lang, 4)
            if 'BaiduAccu' in custom_ocr_engines:
                rec_results_baidu_accu, pic_ocr_data = get_ocr_data('baidu_accu', pic_ocr_data, img_np, media_lang, 4)
            ocr_data[pic_locate] = pic_ocr_data

            lines_tesseract = tesseract2text(tess_zipped_data)
            tesseract_text = lf.join(lines_tesseract)
            lines_vision = rec2text(rec_results_vision)
            vision_text = lf.join(lines_vision)

            refer_ocrs = []
            if 'Baidu' in custom_ocr_engines:
                lines_baidu = [x[-1] for x in rec_results_baidu]
                baidu_text = lf.join(lines_baidu)
                baidu_text_format = better_text(baidu_text, 'baidu')
                refer_ocrs.append(baidu_text_format)
            if 'BaiduAccu' in custom_ocr_engines:
                lines_baidu_accu = [x[-1] for x in rec_results_baidu_accu]
                baidu_accu_text = lf.join(lines_baidu_accu)
                baidu_accu_text_format = better_text(baidu_accu_text, 'baidu_accu')
                refer_ocrs.append(baidu_accu_text_format)
            if 'Paddle' in custom_ocr_engines:
                lines_paddle = [x[-1] for x in rec_results_paddle]
                paddle_text = lf.join(lines_paddle)
                paddle_text_format = better_text(paddle_text, 'paddle')
                refer_ocrs.append(paddle_text_format)
            if 'Easy' in custom_ocr_engines:
                lines_easy = rec2text(rec_results_easy)
                easy_text = lf.join(lines_easy)
                easy_text_format = better_text(easy_text, 'easy')
                refer_ocrs.append(easy_text_format)

            tesseract_text_format = better_text(tesseract_text, 'tesseract')
            refer_ocrs.append(tesseract_text_format)
            vision_text_format = better_text(vision_text, 'vision')
            # 优化 'worlds' 和 'its'
            if 'worlds' in vision_text_format.lower() or 'its' in vision_text_format.lower():
                vision_text_format = better_apostrophe(vision_text_format)
            vision_text_format = fix_w_tess(vision_text_format, tesseract_text_format)
            vision_text_format = better_punct(vision_text_format, refer_ocrs)

            # ================获取气泡内文字的基本信息================
            raw_min_x, raw_min_y, raw_max_x, raw_max_y, min_x, min_y, max_x, max_y, center_pt = single_cnt.letter_coors
            br_w = raw_max_x - raw_min_x
            br_h = raw_max_y - raw_min_y
            src_font_size = 30
            dst_font_size = 30
            if rec_results_vision:
                line_h_mean = mean([x[3] for x in rec_results_vision])
                src_font_size = int(line_h_mean)
                dst_font_size = round(src_font_size * 1.05 / 5) * 5

            br_area_real = (single_cnt.br_w - 2) * (single_cnt.br_h - 2)
            fulfill_ratio = single_cnt.area / br_area_real
            bubble_shape = '未知'
            if fulfill_ratio >= 0.95:
                bubble_shape = '矩形'
            text_direction = 'Horizontal'
            text_alignment = 'Center'
            font_name = global_font_name
            if bubble_shape == '矩形' and global_rec_font_name:
                font_name = global_rec_font_name

            # 判断cp_bubble是否为列表类型，如果是，表示气泡是渐变色的
            if isinstance(cp_bubble, list):
                b0 = cp_bubble[0].split('-')[0]
                b1 = cp_bubble[1].split('-')[0]
                bubble_color_str = f'{b0}-{b1}'
            elif cp_bubble == '':
                bubble_color_str = ''
            else:
                bubble_color_str = cp_bubble.split('-')[0]
            letter_color_str = cp_letter.split('-')[0]
            bubble_meta_list = [
                f'{raw_min_x},{raw_min_y},{br_w},{br_h}',
                f'S{src_font_size}',
                f'D{dst_font_size}',
                bubble_shape,
                text_direction,
                text_alignment,
                bubble_color_str,  # 白色气泡
                letter_color_str,  # 黑色文字
                font_name,
            ]
            bubble_meta_list = [x for x in bubble_meta_list if x != '']
            bubble_meta_str = '~'.join(bubble_meta_list)

            # ================将图片添加到docx文件中================
            pic_tup = ('picture', img_np, pic_locate, pic_ocr_data)
            pic_results.append(pic_tup)
            # ================将识别出的文字添加到docx文件中================
            para_tup = ('paragraph', bubble_meta_str, tess_zipped_data, rec_results_vision, vision_text_format)
            pic_results.append(para_tup)
    return pic_results


# @logger.catch
def process_para(ocr_doc, pic_result, img_np):
    # ================文本================
    bubble_meta_str, tess_zipped_data, rec_results_vision, vision_text_format = pic_result[1:5]
    vision_lines = vision_text_format.splitlines()
    # ================获取气泡元数据================
    bubble_meta_list = bubble_meta_str.split('~')
    coors_str = bubble_meta_list[0]
    src_font_size = bubble_meta_list[1].removeprefix('S')
    dst_font_size = bubble_meta_list[2].removeprefix('D')
    src_font_size = int(src_font_size)
    dst_font_size = int(dst_font_size)
    bubble_shape = bubble_meta_list[3]
    text_direction = bubble_meta_list[4]
    text_alignment = bubble_meta_list[5]
    bubble_color_str = ''
    if len(bubble_meta_list) == 9:
        bubble_color_str = bubble_meta_list[6]
    letter_color_str = bubble_meta_list[-2]
    letter_font_name = bubble_meta_list[-1]
    # 获取气泡坐标
    coors = coors_str.split(',')
    coors = [int(x) for x in coors]
    min_x, min_y, br_w, br_h = coors

    # 计算中心点坐标
    center_x = min_x + br_w / 2
    center_y = min_y + br_h / 2
    center_pt = (int(center_x), int(center_y))

    color_locate = f'{bubble_color_str}-{letter_color_str}'
    char_area_dict, word_actual_areas_dict = {}, {}
    if color_locate in area_dic:
        char_area_dict, word_actual_areas_dict = area_dic[color_locate]

    # ================添加段落================
    new_para = ocr_doc.add_paragraph()
    # ================气泡数据================
    new_run = new_para.add_run(bubble_meta_str)
    new_run = new_para.add_run(lf)  # 软换行
    # ================每一个对话框================
    tess_zipped_data5 = [x for x in tess_zipped_data if x[0] == 5]
    par_nums = [x[3] for x in tess_zipped_data5]
    par_nums = reduce_list(par_nums)
    par_nums.sort()
    line_nums = [x[4] for x in tess_zipped_data5]
    line_nums = reduce_list(line_nums)
    line_nums.sort()

    line_infos = []
    for par_num in par_nums:
        # 每个段落
        for line_num in line_nums:
            # 每一行
            line_data = [x for x in tess_zipped_data5 if x[3] == par_num and x[4] == line_num]
            line_data.sort(key=lambda x: x[5])
            word_imgs = []
            for l in range(len(line_data)):
                # 每个单词
                word_data = line_data[l]
                level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text = word_data
                word_br = (left, top, width, height)
                br_area = width * height
                # 裁剪出单词
                crop_br = (top, top + height, left, left + width)
                word_img = img_np[top - 1:top + height + 1, left - 1:left + width + 1]
                # 检查图像是否为空或维度是否为0
                if word_img.size == 0 or word_img.shape[0] == 0 or word_img.shape[1] == 0:
                    logger.error(f'{crop_br=}')
                elif text != '':
                    pos_meta = (text, par_num, line_num, word_num, left, top, width, height, conf, word_img)
                    word_imgs.append(pos_meta)
            if word_imgs:
                line_infos.append(word_imgs)

    if len(vision_lines) == len(line_infos):
        # 行数相等时
        for l in range(len(line_infos)):
            # 每一行
            word_imgs = line_infos[l]
            ocr_line = vision_lines[l]
            tess_words = [x[0] for x in word_imgs]
            line_words = ocr_line.split(' ')
            if len(word_imgs) == len(line_words):
                # 词数相等时
                for w in range(len(word_imgs)):
                    # 每一词
                    pos_meta = word_imgs[w]
                    line_word = line_words[w]
                    text, par_num, line_num, word_num, left, top, width, height, conf, word_img = pos_meta
                    word_img_png = current_dir / f'word_{par_num}_{line_num}_{word_num}.png'
                    write_pic(word_img_png, word_img)
                    # 计算裁剪出的单词图片的黑色像素面积
                    # 检查是否为 NumPy 数组，如果是，将其转换为 PIL 图像
                    if isinstance(word_img, ndarray):
                        word_img = fromarray(word_img)
                    # 转换为灰度图像
                    gray_img = word_img.convert('L')
                    # 转换为二值图像
                    binary_img = gray_img.point(lambda x: 0 if x <= bit_thres else 255, '1')
                    # 计算黑色像素的数量（假设黑色像素的值为0）
                    black_px_area = binary_img.histogram()[0]
                    all_chars_present = all(char in char_area_dict for char in line_word)
                    new_run = new_para.add_run(line_word)

                    # 使用正则表达式对line_word进行处理，匹配前面有字母或者后面有字母的"I"
                    line_word = sub(r'(?<=[a-zA-Z])I|I(?=[a-zA-Z])', '|', line_word)
                    if all_chars_present:
                        if line_word in word_actual_areas_dict:
                            word_actual_areas = word_actual_areas_dict[line_word]
                            expect_area = sum(word_actual_areas) / len(word_actual_areas)
                        else:
                            expect_area = sum([char_area_dict[char] for char in line_word])
                        refer_ratio = black_px_area / expect_area
                        if refer_ratio >= bold_thres:
                            logger.debug(f'{refer_ratio=:.2f}')
                            new_run.bold = True
                            new_run.italic = True
                    if w != len(word_imgs) - 1:
                        # 如果不是最后一个词就添加空格
                        new_run = new_para.add_run(' ')
            else:
                new_run = new_para.add_run(f'{ocr_line}')
                logger.error(f'词数不相等,{tess_words=},{line_words=}')
            if l != len(line_infos) - 1:
                new_run = new_para.add_run(lf)
    else:
        new_run = new_para.add_run(vision_text_format)
        logger.error(f'行数不相等,{vision_text_format=}')
    ocr_doc.add_paragraph('')
    return ocr_doc


def most_common_color(image):
    """返回图像中出现最多的颜色"""
    pixels = list(image.getdata())
    most_common_pixel = Counter(pixels).most_common(1)[0][0]
    return most_common_pixel


def add_pad2img(image, padding, bg_color=None):
    """给图像加上指定的padding，并使用指定的背景色或默认背景色"""
    if bg_color is None:
        bg_color = most_common_color(image)

    new_width = image.width + 2 * padding
    new_height = image.height + 2 * padding
    padded_img = Image.new('RGB', (new_width, new_height), bg_color)
    padded_img.paste(image, (padding, padding))
    return padded_img


# @logger.catch
def update_ocr_doc(ocr_doc, pic_results, ocr_yml, page_ind, img_list):
    ocr_data = iload_data(ocr_yml)
    # 示例单词，长度为34
    sep_word = "supercalifragilisticexpialidocious"
    font_color = "black"

    img_stems = [x.stem for x in img_list]
    cpre = common_prefix(img_stems)
    csuf = common_suffix(img_stems)
    img_file = img_list[page_ind]
    simple_stem = img_file.stem.removeprefix(cpre).removesuffix(csuf)
    pic_text_png = auto_subdir / f'{img_folder.name}-1拼接-{simple_stem}.png'
    # 图与图之间的空行（像素）
    all_cropped_imgs = []
    cur_img_np = None
    if pic_results:
        for p, pic_result in enumerate(pic_results):
            result_type = pic_result[0]
            if result_type == 'stem':
                # ================页码================
                pic_stem = pic_result[1]
                if 1 <= stem_level <= 9:
                    ocr_doc.add_heading(pic_stem, level=stem_level)
                else:
                    ocr_doc.add_paragraph(pic_stem)
                ocr_doc.add_paragraph('')
            elif result_type == 'picture':
                # ================图片================
                img_np, pic_locate, pic_ocr_data = pic_result[1:4]
                cur_img_np = img_np
                ocr_data[pic_locate] = pic_ocr_data
                img_pil = fromarray(img_np)
                all_cropped_imgs.append(img_pil)
                # 获取图片的dpi，如果没有dpi信息，则使用默认值
                image_dpi = img_pil.info.get('dpi', (default_dpi, default_dpi))[0]
                with BytesIO() as temp_buffer:
                    img_pil.save(temp_buffer, format=docx_img_format.upper())
                    temp_buffer.seek(0)
                    pic_width_inches = img_pil.width / image_dpi
                    ocr_doc.add_picture(temp_buffer, width=Inches(pic_width_inches))
            elif result_type == 'paragraph':
                ocr_doc = process_para(ocr_doc, pic_result, cur_img_np)
        write_yml(ocr_yml, ocr_data)
        # ================生成并保存长图================
        if all_cropped_imgs:
            # 使用Pillow绘制单词并获取其尺寸
            text_img = Image.new("RGBA", (1000, 1000), rgba_zero)
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((0, 0), sep_word, font=msyh_font20, fill=font_color)

            # 裁剪文本图像
            text_bbox = text_img.getbbox()
            cropped_text_img = text_img.crop(text_bbox)
            word_width, word_height = cropped_text_img.size

            # 考虑最低宽度
            max_bubble_width = max(img.width for img in all_cropped_imgs)
            max_width = max(max_bubble_width, word_width)

            # 考虑高度
            total_img_h = sum(img.height for img in all_cropped_imgs)
            total_spacing = stitch_spacing * 2 * (len(all_cropped_imgs) - 1)
            total_word_h = word_height * (len(all_cropped_imgs) - 1)
            total_height = total_img_h + total_spacing + total_word_h

            long_img = Image.new('RGB', (max_width, total_height), color_white)
            y_offset = 0
            for img in all_cropped_imgs[:-1]:  # 除了最后一个图像
                long_img.paste(img, ((max_width - img.width) // 2, y_offset))
                y_offset += img.height + stitch_spacing
                # 粘贴分隔词图像
                word_x = (max_width - word_width) // 2
                long_img.paste(cropped_text_img, (word_x, y_offset), cropped_text_img)
                y_offset += word_height + stitch_spacing
            # 添加最后一个图像
            long_img.paste(all_cropped_imgs[-1], ((max_width - all_cropped_imgs[-1].width) // 2, y_offset))
            long_img = add_pad2img(long_img, 20, color_white)
            write_pic(pic_text_png, long_img)
            for p, cropped_img in enumerate(all_cropped_imgs):
                img_path = auto_subdir / f"Page{simple_stem}_Bubble{p:02}.jpg"
                write_pic(img_path, cropped_img)
    else:
        logger.error(f'{page_ind=}')
    return ocr_doc, all_cropped_imgs


@timer_decorator
# @logger.catch
def step2_order(img_folder, media_type):
    img_list = get_valid_imgs(img_folder)
    frame_yml = img_folder.parent / f'{img_folder.name}.yml'
    order_yml = img_folder.parent / f'{img_folder.name}-气泡排序.yml'
    cnts_dic_pkl = img_folder.parent / f'{img_folder.name}.pkl'
    frame_data = iload_data(frame_yml)
    order_data = iload_data(order_yml)
    cnts_dic = iload_data(cnts_dic_pkl, mode='pkl')
    # ================气泡蒙版================
    all_masks = get_valid_imgs(img_folder, mode='mask')
    if thread_method == 'queue':
        for i in range(len(img_list)):
            img_file = img_list[i]
            img_file, ordered_cnts = order1pic(img_file, frame_data, order_data, all_masks, media_type)
            if ordered_cnts:
                cnts_dic[img_file] = ordered_cnts
    else:
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            futures = [executor.submit(order1pic, img_file, frame_data, order_data, all_masks, media_type)
                       for img_file in img_list]
            for future in as_completed(futures):
                try:
                    img_file, ordered_cnts = future.result()
                    if ordered_cnts:
                        cnts_dic[img_file] = ordered_cnts
                except Exception as e:
                    printe(e)

    with open(cnts_dic_pkl, 'wb') as f:
        pickle.dump(cnts_dic, f)


# @timer_decorator
# @logger.catch
def step3_OCR(img_folder, media_type, media_lang, vert):
    """
    对给定文件夹中的图像进行OCR识别，并将结果保存到一个docx文件中。

    :param img_folder: 包含要识别的图像的文件夹。
    :param frame_yml: 包含框架配置的YAML文件路径。
    :param media_type: 媒体类型，如'Manga'。
    :param media_lang: 用于OCR的媒体语言。
    :param vertical: 是否进行垂直文本识别。
    :param ocr_doc: 生成的DOCX文档
    """
    # step3_OCR 函数接收一个包含要识别的图像的文件夹、一个包含框架配置的YAML文件路径、媒体类型、OCR识别使用的媒体语言和一个指示是否进行垂直文本识别的布尔值。
    # 然后，对文件夹中的每个图像进行OCR识别，并将识别结果保存到一个docx文件中。
    img_list = get_valid_imgs(img_folder)
    frame_yml = img_folder.parent / f'{img_folder.name}.yml'
    order_yml = img_folder.parent / f'{img_folder.name}-气泡排序.yml'
    ocr_yml = img_folder.parent / f'{img_folder.name}-文字识别.yml'
    frame_data = iload_data(frame_yml)
    order_data = iload_data(order_yml)
    ocr_data = iload_data(ocr_yml)
    # ================气泡蒙版================
    all_masks = get_valid_imgs(img_folder, mode='mask')

    all_pic_results = []
    if thread_method == 'queue':
        for i in range(len(img_list)):
            img_file = img_list[i]
            pic_results = ocr1pic(img_file, frame_data, order_data, ocr_data, all_masks, media_type,
                                  media_lang, vert)
            all_pic_results.append(pic_results)
    else:
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            futures = [
                executor.submit(ocr1pic, img_file, frame_data, order_data, ocr_data, all_masks, media_type,
                                media_lang, vert)
                for img_file in img_list]
            for future in as_completed(futures):
                pic_results = future.result()
                all_pic_results.append(pic_results)

        all_pic_results.sort(key=lambda x: x[0][1])

    ocr_doc = Document()
    for page_ind, pic_results in enumerate(all_pic_results):
        ocr_doc, all_cropped_imgs = update_ocr_doc(ocr_doc, pic_results, ocr_yml, page_ind, img_list)
    return ocr_doc


@timer_decorator
def step4_readable(img_folder):
    """
    该函数用于处理OCR识别后的docx文档，使其更易阅读。它将多行文本的段落转换为单行文本，
    并保留原始格式和样式。

    :param src_docx: OCR识别后的docx文件路径。
    :param img_list: 与docx文件关联的图像列表。
    :param read_docx: 可读化docx文件路径。
    :param read_html: 可读化html文件路径。
    :return: new_doc: 处理后的易阅读的docx文档对象。
    """
    src_docx = img_folder.parent / f'{img_folder.name}-2校对.docx'
    read_docx = img_folder.parent / f'{img_folder.name}-3段落.docx'
    read_html = img_folder.parent / f'{img_folder.name}-3段落.html'
    img_list = get_valid_imgs(img_folder)

    # 打开docx文件
    ocr_doc = Document(src_docx)
    readable_doc = Document()

    # 读取并连接文档中的所有段落文本
    full_text = []
    for para in ocr_doc.paragraphs:
        full_text.append(para.text)

    # 创建一个字典，将图像文件名与其在文档中的位置关联起来
    index_dict = {}
    last_ind = 0
    inds = []
    for i in range(len(img_list)):
        img_file = img_list[i]
        if img_file.stem in full_text[last_ind:]:
            ind = full_text[last_ind:].index(img_file.stem) + last_ind
            index_dict[img_file.stem] = ind
            inds.append(ind)
            last_ind = ind
    inds.append(len(full_text))

    # 为每个图像文件创建一个段落字典
    pin = 0
    para_dic = {}
    for i in range(len(img_list)):
        img_file = img_list[i]
        if img_file.stem in index_dict:
            start_i = inds[pin] + 1
            end_i = inds[pin + 1]
            pin += 1
            para_dic[img_file.stem] = full_text[start_i:end_i]

    # 把每个段落的多行文本改成单行文本，同时保留原始格式和样式
    for p in range(len(ocr_doc.paragraphs)):
        paragraph = ocr_doc.paragraphs[p]
        # 去除图片名称所在的行
        if p not in inds:
            para_text = paragraph.text
            para_lines = para_text.splitlines()
            if len(para_lines) >= 2:
                meta_line = para_lines[0]
                valid_para_lines = para_lines[1:]

                # 将每句话首字母大写
                single_line_text = ' '.join(valid_para_lines)
                # 将文本分割成句子并保留句末标点符号
                sentence_parts = re.split(r'( *[.?!…]+[\'")\]，。？！]* *)', single_line_text)
                # 对每个句子进行首字母大写处理
                capitalized_parts = [x.capitalize() for x in sentence_parts]
                # 将处理后的句子连接成一个字符串
                para_text_line = ''.join(capitalized_parts).strip()

                if para_text_line != '':
                    new_para = readable_doc.add_paragraph()

                    start_pos = 0
                    runs = paragraph.runs

                    for r in range(len(runs)):
                        run = runs[r]
                        run_text_raw = run.text
                        run_text = run_text_raw.replace('\n', ' ')
                        # 删除Meta数据行
                        if r == 0:
                            new_run_text_raw = run_text.removeprefix(meta_line).removeprefix(' ')
                            end_pos = start_pos + len(new_run_text_raw)
                        else:
                            end_pos = start_pos + len(run_text)
                        new_run_text = para_text_line[start_pos:end_pos]
                        start_pos = end_pos

                        new_run = new_para.add_run(new_run_text)
                        new_run.bold = run.bold
                        new_run.italic = run.italic
                        new_run.underline = run.underline
                        new_run.font.strike = run.font.strike
                        new_run.font.highlight_color = run.font.highlight_color
                        new_run.font.color.rgb = run.font.color.rgb

    write_docx(read_docx, readable_doc)
    if read_docx.exists():
        with open(read_docx, 'rb') as docx_file:
            result = convert_to_html(docx_file)
            result_html = result.value
        result_html = result_html.replace(r'</p><p>', '</p>\n<p>')
        write_txt(read_html, result_html)
        # ================检查是否有俄罗斯字母================
        soup = BeautifulSoup(result_html, 'html.parser')
        text = soup.get_text()
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            russian_chars = findall(r'[а-яА-ЯёЁ]', line)
            if search(r'[а-яА-ЯёЁ]', line):
                logger.warning(f"第{line_number}行: {line.strip()}")
                # 查找并显示这一行中的俄罗斯字母
                print(f"{''.join(russian_chars)}")
    return readable_doc


# @logger.catch
@timer_decorator
def step5_google_translate(simple_lines, target_lang):
    """
    将 .docx 文档翻译成指定的目标语言，并将翻译后的文本保存到一个 .txt 文件中。

    :param simple_lines: 要翻译的文本列表
    :param target_lang: 目标语言的代码，例如 'zh-CN' 或 'en'
    """
    chunks = []
    current_chunk = ""

    # 将文本分成多个块，以便在翻译时遵守最大字符数限制
    for line in simple_lines:
        # 检查将当前行添加到当前块后的长度是否超过最大字符数
        if len(current_chunk) + len(line) + 1 > max_chars:  # 加1是为了考虑换行符
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += line + "\n"

    # 添加最后一个块（如果有内容的话）
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # ================分段翻译================
    translated_chunks = []
    # 对每个块使用谷歌翻译 API 进行翻译
    for chunk in chunks:
        translated_chunk = GoogleTranslator(source='auto', target=target_lang).translate(chunk)
        translated_chunks.append(translated_chunk)
    # 将翻译后的块连接成一个字符串
    translated_text = lf.join(translated_chunks)
    return translated_text


@timer_decorator
def fill_textarea_in_browser(browser, input_text, activate_browser):
    """
    使用 AppleScript 在指定浏览器中查找并填充页面上的第一个 textarea 元素，然后模拟回车键的按下。

    :param browser (str): 浏览器名称，支持 'Safari' 和 'Google Chrome'。
    :param input_text (str): 需要填入 textarea 的文本。
    :param activate_browser (bool): 是否激活（前台显示）浏览器。

    :return: AppleScript 执行结果，如果有错误返回 None。
    """
    input_lines = input_text.splitlines()
    # 将输入文本复制到剪贴板
    pyperclip.copy(input_text)

    # 创建一个新的 KeyboardEvent，模拟按下回车键。
    # 'keydown' 是事件类型，表示一个按键被按下。
    # 'key' 和 'code' 属性分别表示按下的按键和按键代码。
    # 'which' 属性表示按键的字符编码。
    # 'bubbles' 和 'cancelable' 属性表示事件是否可以冒泡和被取消。
    press_enter_js = '''
        var event = new KeyboardEvent('keydown', {
            'key': 'Enter',
            'code': 'Enter',
            'which': 13,
            'bubbles': true,
            'cancelable': true
        });
        Array.from(document.querySelectorAll('textarea'))[0].dispatchEvent(event);
    '''
    # 使用 AppleScript 粘贴剪贴板的内容到 textarea
    paste_and_press_enter = f'''
        tell application "System Events"
            key code 9 using command down
            delay 0.5
            key code 36
        end tell
    '''

    # 如果需要激活浏览器，则设置 activate_command 为 "activate"，否则为空字符串。
    activate_command = "activate" if activate_browser else ""

    if len(input_lines) == 1:
        # 单行提问
        if browser == 'Safari':
            apple_script = f'''
            tell application "{browser}"
                {activate_command}
                do JavaScript "Array.from(document.querySelectorAll('textarea'))[0].value = '{input_text}'; {press_enter_js}" in front document
            end tell
            '''
        elif browser == 'Google Chrome':
            apple_script = f'''
            tell application "{browser}"
                {activate_command}
                set js_code to "Array.from(document.querySelectorAll('textarea'))[0].value = '{input_text}'; {press_enter_js}"
                execute active tab of front window javascript js_code
            end tell
            '''
        else:
            print(f"Error: Unsupported browser {browser}.")
            return None
    else:
        # 多行提问
        if browser == 'Safari':
            apple_script = f'''
            tell application "{browser}"
                {activate_command}
                do JavaScript "Array.from(document.querySelectorAll('textarea'))[0].focus();" in front document
                {paste_and_press_enter}
            end tell
            '''
        elif browser == 'Google Chrome':
            apple_script = f'''
            tell application "{browser}"
                {activate_command}
                set js_code to "Array.from(document.querySelectorAll('textarea'))[0].focus();"
                execute active tab of front window javascript js_code
                {paste_and_press_enter}
            end tell
            '''
        else:
            print(f"Error: Unsupported browser {browser}.")
            return None
    return run_apple_script(apple_script)


def get_QA(browser):
    """
    从当前浏览器中提取问答对，并进行格式化和清理。

    :param browser: 当前使用的浏览器名称
    :return: 包含已清理和格式化问答对的列表
    """
    # ================保存当前浏览器内容================
    chatgpt_html = save_from_browser(browser)
    chatgpt_text = read_txt(chatgpt_html)
    # ================解析当前浏览器内容================
    soup = BeautifulSoup(chatgpt_text, 'html.parser')
    # 删除所有不需要的标签，如 meta、script、link 和 style
    extra_tags = [
        'meta',
        'script',
        'link',
        'style',
    ]
    for extra_tag in extra_tags:
        for meta in soup.find_all(extra_tag):
            meta.decompose()

    # 查找并删除具有特定 class 的 <div> 标签
    extra_classes = [
        'overflow-x-hidden',
        'bottom-0',
    ]
    for extra_class in extra_classes:
        for div in soup.find_all('div', class_=extra_class):
            div.decompose()

    # 格式化HTML并去除多余的空行
    pretty_html = soup.prettify()
    simple_chatgpt_text = '\n'.join([line for line in pretty_html.splitlines() if line.strip()])
    simple_chatgpt_html = chatgpt_html.parent / f'{chatgpt_html.stem}_simple.html'
    write_txt(simple_chatgpt_html, simple_chatgpt_text)

    # ================进行问答解析================
    simple_soup = BeautifulSoup(simple_chatgpt_text, 'html.parser')
    target_divs = []
    class_ = "w-[calc(100%-50px)]"
    divs = simple_soup.find_all('div', class_=class_)
    for div in divs:
        raw_div = deepcopy(div)
        raw_div_str = str(raw_div)
        # 根据文本特征判断发送者身份
        if 'dark:prose-invert light' in raw_div_str:
            text_role = 'chatGPT'
        else:
            text_role = '用户'

        # 删除最外层的div标签
        for _ in range(4):
            if isinstance(div, Tag) and div.name == 'div':
                for child in div.children:
                    if not isinstance(child, NavigableString):
                        div = child
                        break

        # 删除不需要的外部 div 标签，并提取目标 div
        if isinstance(div, Tag) and div.name == 'div' and len(div.contents) == 1:
            # 删除最外层的 div 标签
            div = div.contents[0].strip()
            target_div = div
        else:
            # 去掉底部的 button 和 svg 标签
            for tag in reversed(div.find_all(['button', 'svg'])):
                tag.extract()

            target_div = div.prettify().strip()
            # 如果包含不符合规定的内容，使用原始 div
            if 'This content may violate our' in target_div:
                target_div = raw_div.prettify().strip()
        # logger.warning(text_role)
        # logger.info(target_div)
        target_divs.append(target_div)
    return target_divs


# @logger.catch
# @timer_decorator
def step5_chatgpt_translate(read_html, raw_html, target_lang):
    """
    将 .docx 文档翻译成指定的目标语言，并将翻译后的文本保存到一个 .txt 文件中。

    :param read_html: 要翻译的 .html 文档的路径
    :param target_lang: 目标语言的代码，例如 'zh-CN' 或 'en'
    :param img_folder: 图片文件夹的路径
    """
    html_text = read_txt(read_html)
    html_lines = html_text.splitlines()

    # ================对网页内容进行分段================
    split_lines = []
    current_lines = []
    current_line_count = 0
    current_char_count = 0
    for line in html_lines:
        line_length = len(line)
        if current_line_count + 1 <= gpt_line_max and current_char_count + line_length <= gpt_char_max:
            current_lines.append(line)
            current_line_count += 1
            current_char_count += line_length
        else:
            split_lines.append(current_lines)
            current_lines = [line]
            current_line_count = 1
            current_char_count = line_length

    if current_lines:
        split_lines.append(current_lines)

    activate_browser = True
    target_divs = get_QA(browser)
    # ================添加提示词================
    sleep_time = sleep_minute * 60
    for s in range(len(split_lines)):
        current_lines = split_lines[s]
        current_text = lf.join(current_lines)
        full_prompt = f'{prompt_prefix}{lf}```html{lf}{current_text}{lf}```'
        if full_prompt not in target_divs and do_automate:
            logger.warning(f'{s=}, {len(current_lines)=}, {len(current_text)=}')
            logger.info(full_prompt)
            fill_textarea_in_browser(browser, full_prompt, activate_browser)
            if s != len(split_lines):
                # 等待回答完成
                sleep(sleep_time)
            else:
                # 最后一次等待时间为一半
                sleep(sleep_time / 2)

    logger.warning(f'chatGPT翻译完成, {len(split_lines)=}')

    target_divs = get_QA(browser)
    unescaped_texts = []
    for s in range(len(split_lines)):
        current_lines = split_lines[s]
        current_text = lf.join(current_lines)
        full_prompt = f'{prompt_prefix}{lf}```html{lf}{current_text}{lf}```'
        index_user = target_divs.index(full_prompt)
        index_chatgpt = index_user + 1
        gpt_html = target_divs[index_chatgpt]
        # logger.info(gpt_html)

        gpt_soup = BeautifulSoup(gpt_html, 'html.parser')
        # 查找 code 标签
        code_tag = gpt_soup.find('code')
        if code_tag is None:
            print()
        # 获取 code 标签的文本内容
        code_text = code_tag.get_text()
        # 对文本内容进行反向转义
        unescaped_text = unescape(code_text)
        # 输出结果
        unescaped_texts.append(unescaped_text.strip())
    dst_html_text = lf.join(unescaped_texts)
    print(dst_html_text)
    write_txt(raw_html, dst_html_text)
    return dst_html_text


def html2docx(html_text, new_para):
    # 生成一个唯一的临时文件名
    temp_filename = f"{uuid4()}.docx"

    # 将HTML转换为docx并将其保存到磁盘上
    convert_text(html_text, 'docx', format='html', outputfile=temp_filename, extra_args=['--wrap=none'])

    # 从磁盘读取生成的docx文件
    with open(temp_filename, 'rb') as f:
        temp_doc = Document(f)

    # 将转换后的docx内容添加到新段落
    for para in temp_doc.paragraphs:
        for run in para.runs:
            new_run = new_para.add_run(run.text)
            new_run.bold = run.bold
            new_run.italic = run.italic
            new_run.underline = run.underline
            new_run.font.size = run.font.size
            new_run.font.name = run.font.name

    # 删除临时文件
    os.remove(temp_filename)


# 将所有英文标点替换为对应的中文标点
punct_map = {
    ',': '，',
    '.': '。',
    '?': '？',
    '!': '！',
    # ':': '：',
    # ';': '；',
}


def get_innermost_tag(tag):
    """递归地获取最内层的标签"""
    if tag.contents and isinstance(tag.contents[-1], Tag):
        return get_innermost_tag(tag.contents[-1])
    return tag


# @logger.catch
@timer_decorator
def get_dst_doc(src_docx, img_list, raw_html, dst_html):
    # 打开docx文件
    ocr_doc = Document(src_docx)
    new_doc = Document()
    dst_html_text = read_txt(raw_html)
    dst_html_text = dst_html_text.replace('…', '...')
    dst_html_text = sub(r'\.{2,}', '…', dst_html_text)
    # 去除中文句子中的空格
    dst_html_text = sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', dst_html_text)
    for eng_punc, chi_punc in punct_map.items():
        dst_html_text = dst_html_text.replace(eng_punc, chi_punc)
    # 去除前后都不是英文单词的空格
    dst_html_text = sub(r'(?<![a-zA-Z])\s|\s(?![a-zA-Z])', '', dst_html_text)

    soup = BeautifulSoup(dst_html_text, 'html.parser')
    chinese_punctuations = "，。？！：；（）【】｛｝《》…“‘”’"
    for p in soup.find_all('p'):
        for idx, child in enumerate(p.contents):
            if (isinstance(child, NavigableString) and child[0] in chinese_punctuations and
                    child.parent.name == 'p'):  # 确保该子节点是<p>标签的直接文本内容
                # 如果当前子节点是字符串并且以中文标点开头
                if idx > 0:  # 确保它不是第一个子节点
                    prev_sibling = p.contents[idx - 1]
                    if isinstance(prev_sibling, Tag) and prev_sibling.name in ['strong', 'em']:
                        innermost_tag = get_innermost_tag(prev_sibling)
                        if innermost_tag.string and innermost_tag.string[-1] != child[0]:
                            innermost_tag.string += child[0]
                            child.replace_with(child[1:])

    # 更新dst_html_text为soup的HTML内容
    dst_html_text = str(soup)
    dst_html_text = dst_html_text.replace('</p><p>', '</p>\n<p>')
    write_txt(dst_html, dst_html_text)
    dst_html_lines = dst_html_text.splitlines()

    img_stems = [x.stem for x in img_list]
    cpre = common_prefix(img_stems)
    csuf = common_suffix(img_stems)

    # 读取并连接文档中的所有段落文本
    full_text = []
    for para in ocr_doc.paragraphs:
        full_text.append(para.text)

    # 创建一个字典，将图像文件名与其在文档中的位置关联起来
    index_dict = {}
    last_ind = 0
    inds = []
    for i in range(len(img_list)):
        img_file = img_list[i]
        if img_file.stem in full_text[last_ind:]:
            ind = full_text[last_ind:].index(img_file.stem) + last_ind
            index_dict[img_file.stem] = ind
            inds.append(ind)
            last_ind = ind
    inds.append(len(full_text))

    dst_pin = 0
    for p in range(len(ocr_doc.paragraphs)):
        paragraph = ocr_doc.paragraphs[p]
        if p in inds:
            # 图片名称所在的行
            pic_name = paragraph.text
            simple_stem = pic_name.removeprefix(cpre).removesuffix(csuf)
            if p != 0:
                new_para = new_doc.add_paragraph('')
            new_para = new_doc.add_paragraph(simple_stem)
        else:
            para_text = paragraph.text
            para_lines = para_text.splitlines()
            if len(para_lines) >= 2:
                meta_line = para_lines[0]
                valid_para_lines = para_lines[1:]
                para_text_line = lf.join(valid_para_lines)
                logger.info(f'[{len(para_lines)}]{para_text_line=}')
                if para_text_line != '':
                    # 放上对应的翻译
                    trans_html_line = dst_html_lines[dst_pin]
                    # 将HTML文本转换为docx格式
                    new_para = new_doc.add_paragraph()
                    html2docx(trans_html_line, new_para)
                    dst_pin += 1
    return new_doc


def step5_translate(img_folder):
    src_docx = img_folder.parent / f'{img_folder.name}-2校对.docx'
    read_html = img_folder.parent / f'{img_folder.name}-3段落.html'
    dst_docx = img_folder.parent / f'{img_folder.name}-4翻译.docx'
    raw_html = img_folder.parent / f'{img_folder.name}-4原翻.html'
    dst_html = img_folder.parent / f'{img_folder.name}-4翻译.html'
    googletrans_txt = img_folder.parent / f'{img_folder.name}-9-api-谷歌翻译.txt'
    img_list = get_valid_imgs(img_folder)
    if not read_html.exists():
        readable_doc = step4_readable(img_folder)
    if read_html.exists():
        read_html_text = read_txt(read_html)
        soup = BeautifulSoup(read_html_text, 'html.parser')
        text = soup.get_text()
        simple_lines = text.splitlines()
        read_html_mtime = getmtime(read_html)
        if not googletrans_txt.exists():
            # ================如有更新则更新谷歌翻译================
            logger.debug('not googletrans_txt.exists()')
            translated_text = step5_google_translate(simple_lines, target_lang)
            write_txt(googletrans_txt, translated_text)
        else:
            # 获取文件的修改时间
            googletrans_txt_mtime = getmtime(googletrans_txt)
            # 比较修改时间
            if read_html_mtime > googletrans_txt_mtime:
                # ================如有更新则更新谷歌翻译================
                logger.debug('read_html_mtime > googletrans_txt_mtime')
                translated_text = step5_google_translate(simple_lines, target_lang)
                write_txt(googletrans_txt, translated_text)
            else:
                # ================否则进行GPT4翻译================
                logger.debug('进行GPT4翻译')
                if not raw_html.exists():
                    step5_chatgpt_translate(read_html, raw_html, target_lang)
                dst_doc = get_dst_doc(src_docx, img_list, raw_html, dst_html)
                write_docx(dst_docx, dst_doc)


@logger.catch
def folder_proc(img_folder, step_str, image_inds):
    frame_yml = img_folder.parent / f'{img_folder.name}.yml'
    ocr_docx = img_folder.parent / f'{img_folder.name}-1识别.docx'
    src_docx = img_folder.parent / f'{img_folder.name}-2校对.docx'
    read_docx = img_folder.parent / f'{img_folder.name}-3段落.docx'
    read_html = img_folder.parent / f'{img_folder.name}-3段落.html'
    dst_docx = img_folder.parent / f'{img_folder.name}-4翻译.docx'
    raw_html = img_folder.parent / f'{img_folder.name}-4原翻.html'
    dst_html = img_folder.parent / f'{img_folder.name}-4翻译.html'
    cut_txt = img_folder.parent / f'{img_folder.name}-5断句.txt'
    lettering_txt = img_folder.parent / f'{img_folder.name}-6填字.txt'
    mark_txt = img_folder.parent / f'{img_folder.name}-7标记.txt'
    googletrans_txt = img_folder.parent / f'{img_folder.name}-9-api-谷歌翻译.txt'
    img_list = get_valid_imgs(img_folder)
    auto_subdir = Auto / img_folder.name
    make_dir(auto_subdir)

    media_type = img_folder.parent.name.removesuffix('Process')

    if '0' in step_str:
        frame_data_sorted = step0_analyze_frames(img_folder, frame_yml, media_type, auto_subdir, image_inds)
    if '1' in step_str:
        auto_all_masks = step1_analyze_bubbles(img_folder, media_type, auto_subdir)
    if '2' in step_str:
        step2_order(img_folder, media_type)
    if '3' in step_str:
        ocr_doc = step3_OCR(img_folder, media_type, media_lang, vert)
        write_docx(ocr_docx, ocr_doc)
    if '4' in step_str:
        readable_doc = step4_readable(img_folder)
    if '5' in step_str:
        step5_translate(img_folder)


def z():
    pass


if __name__ == "__main__":
    Comic = homedir / 'Comic'
    Hanhua = Comic / '汉化'

    MomoHanhua = DOCUMENTS / '默墨汉化'
    Auto = MomoHanhua / 'Auto'
    Log = MomoHanhua / 'Log'
    FontLibrary = MomoHanhua / 'FontLibrary'
    ComicProcess = MomoHanhua / 'ComicProcess'
    MangaProcess = MomoHanhua / 'MangaProcess'
    ManhuaProcess = MomoHanhua / 'ManhuaProcess'

    MomoYolo = DOCUMENTS / '默墨智能'
    Storage = MomoYolo / 'Storage'
    ChatGPT = MomoYolo / 'ChatGPT'

    make_dir(Comic)
    make_dir(Hanhua)

    make_dir(MomoHanhua)
    make_dir(Auto)
    make_dir(Log)
    make_dir(ComicProcess)
    make_dir(MangaProcess)
    make_dir(ManhuaProcess)

    make_dir(MomoYolo)
    make_dir(Storage)
    make_dir(ChatGPT)

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

    CTD_onnx = Storage / 'comictextdetector.pt.onnx'
    CTD_model = None
    uoln = None

    # ================选择语言================
    # 不支持使用软件期间重选语言
    # 因为界面代码是手写的，没有 retranslateUi
    # lang_code = 'en_US'
    lang_code = 'zh_CN'
    # lang_code = 'zh_TW'
    # lang_code = 'ja_JP'
    qm_path = UserDataFolder / f'{APP_NAME}_{lang_code}.qm'

    do_qt_translate = False
    do_qt = False
    do_dev_folder = False
    do_dev_pic = False
    do_requirements = False
    do_structure = False
    do_roi = False

    mode_list = [
        'do_qt_translate',
        'do_qt',
        'do_dev_folder',
        'do_dev_pic',
        'do_requirements',
        'do_structure',
        'do_roi',
    ]


    def steps():
        pass


    system_fonts = font_manager.findSystemFonts()
    system_font_dic = {}
    system_fonts = sorted(system_fonts)
    for i in range(len(system_fonts)):
        font = system_fonts[i]
        font_path = Path(font)
        if font_path.exists():
            system_font_dic[font_path.name] = font_path.as_posix()

    app_config = AppConfig.load()

    do_mode = app_config.config_data['do_mode']
    step_str = app_config.config_data['step_str']
    folder_name = app_config.config_data['folder_name']
    area_folder_name = app_config.config_data['area_folder_name']
    browser = app_config.config_data['browser']
    thumb_size = app_config.config_data['thumb_size']
    window_size = app_config.config_data['window_size']
    if ',' in window_size:
        sizes = window_size.split(',')
        sizes = [int(x) for x in sizes]
        window_w, window_h = sizes
    else:
        window_w = window_h = window_size
    WBLDN_padding = app_config.config_data['WBLDN_padding']
    if ',' in WBLDN_padding:
        paddings = WBLDN_padding.split(',')
        paddings = [int(x) for x in paddings]
        white_padding, black_padding, light_padding, dark_padding, normal_padding = paddings
    else:
        white_padding = black_padding = light_padding = dark_padding = normal_padding = WBLDN_padding
    stem_level = app_config.config_data['stem_level']
    media_lang = app_config.config_data['media_lang']
    target_lang = app_config.config_data['target_lang']
    vert = False  # 设置为True以识别竖排文本
    if media_lang == 'Japanese':
        vert = True
    default_dpi = app_config.config_data['default_dpi']
    thread_method = app_config.config_data['thread_method']
    pic_thread_method = app_config.config_data['pic_thread_method']
    if use_torch:
        pic_thread_method = 'queue'

    lettering = app_config.config_data['lettering']
    global_font_size = lettering['global_font_size']
    global_font_name = lettering['global_font_name']
    global_rec_font_name = lettering['global_rec_font_name']
    global_spacing = lettering['global_spacing']

    font_dic = app_config.config_data['font_dic']

    chatgpt = app_config.config_data['chatgpt']
    do_automate = chatgpt['do_automate']
    sleep_minute = chatgpt['sleep_minute']
    gpt_line_max = chatgpt['gpt_line_max']
    gpt_char_max = chatgpt['gpt_char_max']

    long_pic = app_config.config_data['long_pic']
    single_width_min = long_pic['single_width_min']
    group_len = long_pic['group_len']

    frame_settings = app_config.config_data['frame_settings']
    frame_color = frame_settings['frame_color']
    min_frame_length = frame_settings['min_frame_length']
    edge_size = frame_settings['edge_size']
    edge_ratio = frame_settings['edge_ratio']
    grid_width_min = frame_settings['grid_width_min']
    grid_height_min = frame_settings['grid_height_min']
    max_white_ratio = frame_settings['max_white_ratio']
    do_add_frame = frame_settings['do_add_frame']
    frame_thres = frame_settings['frame_thres']
    min_width = frame_settings['min_width']
    gaps = frame_settings['gaps']
    if isinstance(gaps, int):
        gaps = [gaps] * 4
    check_second_color = frame_settings['check_second_color']
    check_more_frame_color = frame_settings['check_more_frame_color']
    y_first = frame_settings['y_first']
    fully_framed = frame_settings['fully_framed']

    bubble_condition = app_config.config_data['bubble_condition']
    area_range = bubble_condition['area_range']
    perimeter_range = bubble_condition['perimeter_range']
    thickness_range = bubble_condition['thickness_range']
    br_w_range = bubble_condition['br_w_range']
    br_h_range = bubble_condition['br_h_range']
    br_wh_range = bubble_condition['br_wh_range']
    br_wnh_range = bubble_condition['br_wnh_range']
    br_w_ratio_range = bubble_condition['br_w_ratio_range']
    br_h_ratio_range = bubble_condition['br_h_ratio_range']
    br_ratio_range = bubble_condition['br_ratio_range']
    portion_ratio_range = bubble_condition['portion_ratio_range']
    area_perimeter_ratio_range = bubble_condition['area_perimeter_ratio_range']
    bubble_px_range = bubble_condition['bubble_px_range']
    letter_px_range = bubble_condition['letter_px_range']
    bubble_px_ratio_range = bubble_condition['bubble_px_ratio_range']
    letter_px_ratio_range = bubble_condition['letter_px_ratio_range']
    BnL_px_ratio_range = bubble_condition['BnL_px_ratio_range']
    mask_px_ratio_range = bubble_condition['mask_px_ratio_range']
    edge_px_count_range = bubble_condition['edge_px_count_range']
    letter_area_range = bubble_condition['letter_area_range']
    textblock_area_range = bubble_condition['textblock_area_range']
    letter_cnts_range = bubble_condition['letter_cnts_range']
    textblock_letters_range = bubble_condition['textblock_letters_range']
    note_area_range = bubble_condition['note_area_range']
    max_note_dist = bubble_condition['max_note_dist']
    intersect_ratio = bubble_condition['intersect_ratio']
    seg_line_range = bubble_condition['seg_line_range']
    angle_px_step = bubble_condition['angle_px_step']
    if ',' in angle_px_step:
        steps = angle_px_step.split(',')
        steps = [int(x) for x in steps]
        angle_step, px_step = steps
    else:
        angle_step = px_step = angle_px_step
    WBN_tolerance = bubble_condition['WBN_tolerance']
    if ',' in WBN_tolerance:
        tolerances = WBN_tolerance.split(',')
        tolerances = [int(x) for x in tolerances]
        white_tolerance, black_tolerance, normal_tolerance = tolerances
    else:
        white_tolerance = black_tolerance = normal_tolerance = WBN_tolerance

    area_min, area_max = parse_range(area_range)
    perimeter_min, perimeter_max = parse_range(perimeter_range)
    thickness_min, thickness_max = parse_range(thickness_range)
    br_w_min, br_w_max = parse_range(br_w_range)
    br_h_min, br_h_max = parse_range(br_h_range)
    br_wh_min, br_wh_max = parse_range(br_wh_range)
    br_wnh_min, br_wnh_max = parse_range(br_wnh_range)
    br_w_ratio_min, br_w_ratio_max = parse_range(br_w_ratio_range)
    br_h_ratio_min, br_h_ratio_max = parse_range(br_h_ratio_range)
    br_ratio_min, br_ratio_max = parse_range(br_ratio_range)
    portion_ratio_min, portion_ratio_max = parse_range(portion_ratio_range)
    area_perimeter_ratio_min, area_perimeter_ratio_max = parse_range(area_perimeter_ratio_range)
    bubble_px_min, bubble_px_max = parse_range(bubble_px_range)
    letter_px_min, letter_px_max = parse_range(letter_px_range)
    bubble_px_ratio_min, bubble_px_ratio_max = parse_range(bubble_px_ratio_range)
    letter_px_ratio_min, letter_px_ratio_max = parse_range(letter_px_ratio_range)
    BnL_px_ratio_min, BnL_px_ratio_max = parse_range(BnL_px_ratio_range)
    mask_px_ratio_min, mask_px_ratio_max = parse_range(mask_px_ratio_range)
    edge_px_count_min, edge_px_count_max = parse_range(edge_px_count_range)
    letter_area_min, letter_area_max = parse_range(letter_area_range)
    textblock_area_min, textblock_area_max = parse_range(textblock_area_range)
    letter_cnts_min, letter_cnts_max = parse_range(letter_cnts_range)
    textblock_letters_min, textblock_letters_max = parse_range(textblock_letters_range)
    note_area_min, note_area_max = parse_range(note_area_range)
    seg_line_min, seg_line_max = parse_range(seg_line_range)

    char_condition = app_config.config_data['char_condition']
    px_area_range = char_condition['px_area_range']
    cnt_area_range = char_condition['cnt_area_range']
    thick_range = char_condition['thick_range']
    brw_range = char_condition['brw_range']
    brh_range = char_condition['brh_range']

    px_area_min, px_area_max = parse_range(px_area_range)
    cnt_area_min, cnt_area_max = parse_range(cnt_area_range)
    thick_min, thick_max = parse_range(thick_range)
    brw_min, brw_max = parse_range(brw_range)
    brh_min, brh_max = parse_range(brh_range)

    bubble_recognize = app_config.config_data['bubble_recognize']
    WLB_ext_px = bubble_recognize['WLB_ext_px']
    max_height_diff = bubble_recognize['max_height_diff']
    if ',' in str(WLB_ext_px):
        ext_pxs = WLB_ext_px.split(',')
        ext_pxs = [int(x) for x in ext_pxs]
        word_ext_px, line_ext_px, block_ext_px = ext_pxs
    else:
        word_ext_px = line_ext_px = block_ext_px = int(WLB_ext_px)
    max_font_size = bubble_recognize['max_font_size']
    kernel_depth = bubble_recognize['kernel_depth']
    block_ratio = bubble_recognize['block_ratio']
    bubble_alpha = bubble_recognize['bubble_alpha']
    bg_alpha = bubble_recognize['bg_alpha']
    WLB_alpha = bubble_recognize['WLB_alpha']
    if ',' in WLB_alpha:
        alphas = WLB_alpha.split(',')
        alphas = [int(x) / 100 for x in alphas]
        textword_alpha, textline_alpha, textblock_alpha = alphas
    else:
        textword_alpha = textline_alpha = textblock_alpha = int(WLB_alpha) / 100
    padding = bubble_recognize['padding']

    bubble_seg = app_config.config_data['bubble_seg']
    all_caps = bubble_seg['all_caps']
    adapt_big_letter = bubble_seg['adapt_big_letter']
    check_note = bubble_seg['check_note']
    check_dots = bubble_seg['check_dots']
    check_intact = bubble_seg['check_intact']
    use_pivot_split = bubble_seg['use_pivot_split']

    ocr_settings = app_config.config_data['ocr_settings']
    bold_thres = ocr_settings['bold_thres']
    y_thres = ocr_settings['y_thres']
    bit_thres = ocr_settings['bit_thres']
    print_tables = ocr_settings['print_tables']
    init_ocr = ocr_settings['init_ocr']
    stitch_spacing = ocr_settings['stitch_spacing']

    baidu_ocr = app_config.config_data['baidu_ocr']
    obd_app_id = baidu_ocr['APP_ID']
    obd_app_key = baidu_ocr['API_KEY']
    obd_secret_key = baidu_ocr['SECRET_KEY']
    aip_client = None
    if obd_app_id and obd_app_key and obd_secret_key:
        aip_client = AipOcr(obd_app_id, obd_app_key, obd_secret_key)

    custom_ocr_engines = app_config.config_data['custom_ocr_engines']
    prompt_prefix = app_config.config_data['prompt_prefix']
    img_ind = app_config.config_data['img_ind']
    grid_ratio_dic = app_config.config_data['grid_ratio_dic']
    color_pattern_dic = app_config.config_data['color_pattern_dic']

    font_meta_csv = UserDataFolder / f'字体信息_{processor()}_{ram}GB.csv'
    font_head = ['字体文件名', '字体名', '字体PostScript名']
    font_meta_list = update_font_metadata(font_meta_csv, font_head)

    font_filename = font_dic.get('微软雅黑')
    msyh_font_ttc = FontLibrary / font_filename
    msyh_font20 = ImageFont.truetype(msyh_font_ttc.as_posix(), 20)
    msyh_font30 = ImageFont.truetype(msyh_font_ttc.as_posix(), 30)
    msyh_font60 = ImageFont.truetype(msyh_font_ttc.as_posix(), 60)
    msyh_font100 = ImageFont.truetype(msyh_font_ttc.as_posix(), 100)

    for mode in mode_list:
        globals()[mode] = (do_mode == mode)

    img_folder = ComicProcess / folder_name
    area_img_folder = ComicProcess / area_folder_name
    # img_folder = MangaProcess / folder_name
    # area_img_folder = MangaProcess / area_folder_name

    area_yml = area_img_folder.parent / f'{area_img_folder.name}-文字面积.yml'

    cut_line_txt = UserDataFolder / '断行模式.txt'
    cut_line_text = read_txt(cut_line_txt)
    cut_line_list = cut_line_text.splitlines()
    cut_models = []
    for c in range(len(cut_line_list)):
        cut_model_str = cut_line_list[c]
        cut_model = cut_model_str.split('|')
        model_cuts = cut_model[-1].split(',')
        model_cuts = [int(x) for x in model_cuts]
        cut_model = [int(cut_model[0]), int(cut_model[1]), tuple(model_cuts)]
        cut_models.append(cut_model)

    auto_subdir = Auto / img_folder.name
    make_dir(auto_subdir)
    media_type = img_folder.parent.name.removesuffix('Process')
    img_list = get_valid_imgs(img_folder)
    img_stems = [x.stem for x in img_list]
    cpre = common_prefix(img_stems)
    csuf = common_suffix(img_stems)
    all_masks = get_valid_imgs(img_folder, mode='mask')

    # ================获取漫画名================
    series4file = folder_name
    p_issue_w_dot = re.compile(r'(.+?)(?!\d) (\d{2,5})', I)
    M_issue_w_dot = p_issue_w_dot.match(folder_name)
    if M_issue_w_dot:
        series4file = M_issue_w_dot.group(1)
        issue = M_issue_w_dot.group(2)

    # ================气泡颜色模式================
    color_patterns_raw = app_config.config_data.get('color_patterns', [])
    media_type_color_patterns = color_pattern_dic.get(media_type, [])
    series4file_color_patterns = color_pattern_dic.get(series4file, [])
    if color_patterns_raw is None:
        color_patterns_raw = []
    if media_type_color_patterns:
        color_patterns_raw.extend(media_type_color_patterns)
    if series4file_color_patterns:
        color_patterns_raw.extend(series4file_color_patterns)
    color_patterns_raw = reduce_list(color_patterns_raw)

    color_patterns = []
    for c in range(len(color_patterns_raw)):
        cp_str = color_patterns_raw[c]
        cp_bubble, par, cp_letter = cp_str.partition('~')
        if '=' in cp_bubble:
            cp1, par, cp2 = cp_bubble.partition('=')
            cp_bubble = [cp1, cp2]
        if '=' in cp_letter:
            cp3, par, cp4 = cp_letter.partition('=')
            cp_letter = [cp3, cp4]
        if cp_letter == '':
            cp_bubble, cp_letter = cp_letter, cp_bubble
        color_pattern = [cp_bubble, cp_letter]

        color_patterns.append(color_pattern)
    logger.debug(f'{color_patterns=}')

    if use_torch and CTD_onnx.exists():
        CTD_model = dnn.readNetFromONNX(CTD_onnx.as_posix())
        uoln = CTD_model.getUnconnectedOutLayersNames()

    if '3' in step_str or '9' in step_str:
        en_reader = None
        en_ocr = None
        if init_ocr:
            en_reader = Reader(['en'])
            en_ocr = PaddleOCR(use_gpu=False, lang='en')

    logger.warning(f'{do_mode=}, {thread_method=}, {pic_thread_method=}')
    if do_qt:
        appgui = QApplication(sys.argv)
        translator = QTranslator()
        translator.load(str(qm_path))
        QApplication.instance().installTranslator(translator)
        appgui.installTranslator(translator)
        screen = QApplication.primaryScreen()
        if sys.platform == 'darwin':
            # 如果是 MacOS 系统
            scaling_factor = screen.devicePixelRatio()
        else:
            scaling_factor = 1
        scaling_factor_reci = 1 / scaling_factor
        mist_qt(appgui)
    elif do_dev_folder:
        logger.debug(f'{folder_name=}')
        logger.info(f'{step_str=}')
        image_inds = [
            # 7,
        ]
        area_dic = get_area_dic(area_yml)
        folder_proc(img_folder, step_str, image_inds)
    elif do_dev_pic:
        img_file = img_list[img_ind]
        image_raw = imdecode(fromfile(img_file, dtype=uint8), -1)
        ih, iw = image_raw.shape[0:2]

        frame_yml = img_folder.parent / f'{img_folder.name}.yml'
        order_yml = img_folder.parent / f'{img_folder.name}-气泡排序.yml'
        ocr_yml = img_folder.parent / f'{img_folder.name}-文字识别.yml'
        ocr_docx = img_folder.parent / f'{img_folder.name}-1识别.docx'
        src_docx = img_folder.parent / f'{img_folder.name}-2校对.docx'
        read_docx = img_folder.parent / f'{img_folder.name}-3段落.docx'
        read_html = img_folder.parent / f'{img_folder.name}-3段落.html'
        dst_docx = img_folder.parent / f'{img_folder.name}-4翻译.docx'
        raw_html = img_folder.parent / f'{img_folder.name}-4原翻.html'
        dst_html = img_folder.parent / f'{img_folder.name}-4翻译.html'
        cut_txt = img_folder.parent / f'{img_folder.name}-5断句.txt'
        lettering_txt = img_folder.parent / f'{img_folder.name}-6填字.txt'
        mark_txt = img_folder.parent / f'{img_folder.name}-7标记.txt'
        googletrans_txt = img_folder.parent / f'{img_folder.name}-9-api-谷歌翻译.txt'

        img_stems = [x.stem for x in img_list]
        cpre = common_prefix(img_stems)
        csuf = common_suffix(img_stems)
        simple_stems = [x.removeprefix(cpre).removesuffix(csuf) for x in img_stems]

        logger.info(f'{step_str=}')
        media_lang = 'English'
        target_lang = 'zh-CN'
        vert = False  # 设置为True以识别竖排文本
        simple_stem = img_file.stem.removeprefix(cpre).removesuffix(csuf)
        pic_ocr_docx = img_folder.parent / f'{img_folder.name}-1识别-{simple_stem}.docx'
        area_dic = get_area_dic(area_yml)
        if media_lang == 'Japanese':
            vert = True

        frame_data = iload_data(frame_yml)
        if '0' in step_str:
            img_file, frame_grid_strs = analyze1frame(img_file, frame_data, auto_subdir, media_type)
        frame_data = iload_data(frame_yml)
        order_data = iload_data(order_yml)
        ocr_data = iload_data(ocr_yml)
        if '1' in step_str:
            analyze1pic(img_file, frame_data, color_patterns, media_type, auto_subdir)
        if '2' in step_str:
            appgui = QApplication(sys.argv)
            translator = QTranslator()
            translator.load(str(qm_path))
            QApplication.instance().installTranslator(translator)
            appgui.installTranslator(translator)
            screen = QApplication.primaryScreen()
            if sys.platform == 'darwin':
                # 如果是 MacOS 系统
                scaling_factor = screen.devicePixelRatio()
            else:
                scaling_factor = 1
            scaling_factor_reci = 1 / scaling_factor
            order_qt(appgui)
        if '3' in step_str:
            pic_results = ocr1pic(img_file, frame_data, order_data, ocr_data, all_masks, media_type, media_lang,
                                  vert)
            ocr_doc = Document()
            ocr_doc, all_cropped_imgs = update_ocr_doc(ocr_doc, pic_results, ocr_yml, img_ind, img_list)
            write_docx(pic_ocr_docx, ocr_doc)
