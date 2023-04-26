
# ========================standard packages=============================

from ast import Import, ImportFrom, parse, walk
from copy import deepcopy
import getopt
from getpass import getuser
from pathlib import Path
import re
from subprocess import call
import sys
from time import strftime

# ===========================3rdp packages==============================

import cv2
from loguru import logger
from PyQt6.QtWidgets import QApplication
import pkg_resources
from stdlib_list import stdlib_list
import xmltodict

# ==========================custom packages=============================

from config import GlobalConfig
from storage import GlobalStorage

from gui import GUI

from helper import timer_decorator

from ioutils.txt import read_txt, write_txt
from ioutils.csv import iread_csv, write_csv

from utils import is_decimal_or_comma

# ================================参数区================================

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
py_path = Path(__file__).resolve()
py_dev_path = py_path.parent / f'{py_path.stem}_dev.py'

special_keywords = [
    'setIcon', 'icon',
    'setShortcut', 'QKeySequence',
    'QSettings', 'value', 'setValue',
    'triggered',
    'setWindowTitle', 'windowTitle',
]

supported_langs = [
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

edge_size = 10

padding = 10

def is_valid_file(file_path, suffixes):
    if not file_path.is_file():
        return False
    if not file_path.stem.startswith(('~$', '._')):
        if suffixes:
            return file_path.suffix.lower() in suffixes
        else:
            return True
    return False

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

def update_qm():
    src_head = ['Source', '目标语言']

    for i in range(len(supported_langs)):
        lang, en_name, cn_name, self_name = supported_langs[i]

        ts_file = GlobalStorage.GUI_TRANSLATION_FOLDER / f'{GlobalConfig.APP_NAME}_{lang}.ts'
        csv_file = GlobalStorage.GUI_TRANSLATION_FOLDER / f'{GlobalConfig.APP_NAME}_{lang}.csv'

        # 提取可翻译的字符串生成ts文件
        cmd = f'{GlobalStorage.PYLUPDATE_EXE} {py_path.as_posix()} -ts {ts_file.as_posix()}'
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
        cmd = f'{GlobalStorage.LRELEASE_EXE} {ts_file.as_posix()}'
        call(cmd, shell=True)

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
    if float(python_version) < 3.10:
        stdlib_modules = set(stdlib_list(python_version))
    else:
        stdlib_modules = set(sys.stdlib_module_names)

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
    requirements_text = GlobalConfig.LINE_FEED.join(requirements)
    print(requirements_text)

def print_usage():
    print("Usage: main.py -m qt")
    print("")
    print("  -i, --mode=        program mode, qt | dev | req")
    print("      --update_qm    update qm files")
    print("  -v, --version      print version")
    print("  -h, --help         print usage")

def print_version():
    # hardcode or read global config
    print("0.0.1")

def init_logger():
    date_str = strftime('%Y_%m_%d')
    log_path = GlobalStorage.Log / f'日志-{date_str}.log'
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

def run_gui():
    if GlobalStorage.ONNXModel.exists():
        GlobalStorage.DNNNet = cv2.dnn.readNetFromONNX(GlobalStorage.ONNXModel.as_posix())
        GlobalStorage.UOLayerNames = GlobalStorage.DNNNet.getUnconnectedOutLayersNames()

    app = QApplication(sys.argv)
    GUI()
    sys.exit(app.exec())

def enter_dev():
    # TODO
    pass

TRANSLATOR_MODE = ["qt", "dev"]

def initialize():
    GlobalConfig.init_platform()
    GlobalStorage.init_storage()
    init_logger()

# main entry
def run(argv):
    if argv is None:
        argv = sys.argv
    argv = [ n for n in argv ]

    try:
        opts, args = getopt.getopt(argv[1:], 
            '-m:-h-v', 
            ['mode=', 'help', 'version', 'generate_req', 'update_qm'])
    except getopt.GetoptError as err:
        print(err)
        print_usage()
        return -1
    
    initialize()
    
    # default mode
    translator_mode = "qt"

    for opt_name, opt_value in opts:
        if opt_name == '--generate_req':
            generate_requirements(py_path, python_version)
        elif opt_name in ('-h', '--help'):
            print_usage()
        elif opt_name in ('-m', '--mode'):
            if opt_value in TRANSLATOR_MODE:
                translator_mode = opt_value
        elif opt_name == '--update_qm':
            update_qm()
        elif opt_name in ('-v', '--version'):
            print_version()
    

    if translator_mode == 'qt':
        run_gui()
    elif translator_mode == 'dev':
        enter_dev()


if __name__ == '__main__':
    run(sys.argv)