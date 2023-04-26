from easyocr import Reader

from helper import Singleton
from OCR.OCR import OCR

class EasyOCR(Singleton, OCR):
    def __init__(self):
        pass

    def process(self, image, lang, vertical=False):
        reader = Reader([lang], gpu=False)
        result = reader.readtext(image, detail=0, paragraph=True, width_ths=0.8, text_threshold=0.5, slope_ths=0.5,
                                ycenter_ths=0.5)
        recognized_text = ' '.join(result)
        return recognized_text.strip()