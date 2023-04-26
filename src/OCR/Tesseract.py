from pytesseract import image_to_string

from helper import Singleton
from OCR.OCR import OCR

tesseract_language_options = {
    'Chinese Simplified': 'chi_sim',
    'Chinese Traditional': 'chi_tra',
    'English': 'eng',
    'Japanese': 'jpn',
    'Korean': 'kor',
}

class Tesseract(Singleton, OCR):
    def __init__(self):
        pass

    def process(self, image, lang, vertical=False):
        # 获取所选语言的配置
        language_config = tesseract_language_options[lang]
        config = f'-c preserve_interword_spaces=1 --psm 6 -l {language_config}'
        # 如果识别竖排文本
        if vertical:
            config += " -c textord_old_baselines=0"
        if vertical:
            config += " -c textord_old_baselines=0"
        text = image_to_string(image, config=config)
        return text.strip()
