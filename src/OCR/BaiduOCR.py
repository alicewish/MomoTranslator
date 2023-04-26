from base64 import b64encode

from aip import AipOcr
from cv2 import imencode

from helper import Singleton
from OCR.OCR import OCR


baidu_language_options = {
    'Chinese Simplified': 'CHN_ENG',
    'Chinese Traditional': 'CHN_ENG',
    'English': 'ENG',
    'Japanese': 'JPN',
    'Korean': 'KOR',
}

class BaiduOCR(Singleton, OCR):
    def __init__(self):
        self._app_id = 'your_baidu_app_id'
        self._api_key = 'your_baidu_api_key'
        self._secret_key = 'your_baidu_secret_key'

    def initWith(self, app_id, api_key, secret_key):
        # TODO do preprocess
        self._app_id = app_id
        self._api_key = api_key
        self._secret_key = secret_key

    def process(self, image, lang):
        # 初始化百度OCR客户端
        client = AipOcr(self._app_id, self._api_key, self._secret_key)

        # 将图像转换为base64编码
        _, encoded_image = imencode('.png', image)
        image_data = encoded_image.tobytes()
        base64_image = b64encode(image_data)

        # 设置百度OCR识别选项
        options = {'language_type': baidu_language_options[lang]}

        # 调用百度OCR API进行文字识别
        result = client.basicGeneral(base64_image, options)

        # 提取识别结果
        recognized_text = ''
        if 'words_result' in result:
            words = [word['words'] for word in result['words_result']]
            recognized_text = ' '.join(words)

        return recognized_text.strip()