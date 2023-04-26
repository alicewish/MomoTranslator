from helper import Singleton
from OCR.OCR import OCR

vision_language_options = {
    'Chinese Simplified': 'zh-Hans',
    'Chinese Traditional': 'zh-Hant',
    'English': 'en',
    'Japanese': 'ja',
    'Korean': 'ko',
}

class Vision(Singleton, OCR):
    def __init__(self):
        pass

    def process(self, image, lang):
        languages = [vision_language_options[lang]]  # 设置需要识别的语言

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