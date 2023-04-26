from platform import system, uname

class GlobalConfig:
    APP_NAME = 'MomoTranslator'
    MAJOR_VERSION = 1
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    APP_VERSION = f'v{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}'
    APP_AUTHOR = '墨问非名'

    # default utf-8
    SYSTEM = ''
    ENCODING = 'utf-8'
    LINE_FEED = '\n'
    CMCT = 'command'

    def __init__(self) -> None:
        pass

    @staticmethod
    def init_platform():
        s = system()
        u = uname()
        if s == 'Windows':
            GlobalConfig.SYSTEM = 'WINDOWS'
            GlobalConfig.CMCT = 'ctrl'
        elif s == "Linux":
            GlobalConfig.SYSTEM = 'LINUX'
        elif s == "Darwin":
            GlobalConfig.SYSTEM = 'M1'
        elif u.machine in ['x86_64', 'AMD64']:
            GlobalConfig.SYSTEM = 'MAC'
        else:
            GlobalConfig.SYSTEM = 'PI'

        if GlobalConfig.SYSTEM in ['MAC', 'M1']:
            from Quartz import kCGRenderingIntentDefault
            from Quartz.CoreGraphics import CGDataProviderCreateWithData, CGColorSpaceCreateDeviceRGB, CGImageCreate
            from Vision import VNImageRequestHandler, VNRecognizeTextRequest

