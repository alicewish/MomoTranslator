import os
from pathlib import Path
from PIL import ImageFont

from ioutils.f import make_dir

class GlobalStorage:
    homedir = os.path.expanduser("~")
    DOWNLOADS = Path(homedir) / 'Downloads'
    DOCUMENTS = Path(homedir) / 'Documents'

    MomoHanhua = DOCUMENTS / '默墨汉化'
    Auto = MomoHanhua / 'Auto'
    Log = MomoHanhua / 'Log'
    ComicProcess = MomoHanhua / 'ComicProcess'
    MangaProcess = MomoHanhua / 'MangaProcess'
    ManhuaProcess = MomoHanhua / 'ManhuaProcess'

    MomoYolo = DOCUMENTS / '默墨智能'
    Storage = MomoYolo / 'Storage'
    ONNXModel = Storage / 'comictextdetector.pt.onnx'
    DNNNet = None
    UOLayerNames = []

    ROOT = Path(os.getcwd()).parent
    GUI_TRANSLATION_FOLDER = ROOT / 'tr'
    GUI_SETTINGS = MomoHanhua / "GUISettings.ini"

    # For QM generator
    PYLUPDATE_EXE = 'pylupdate6'
    LRELEASE_EXE = 'lrelease'

    # Arial in Windows 10
    font_path = "ARIALUNI.TTF"
    font60 = ImageFont.truetype(font_path, 60)
    font100 = ImageFont.truetype(font_path, 100)

    @classmethod
    def init_storage(cls):
        make_dir(cls.MomoHanhua)
        make_dir(cls.Auto)
        make_dir(cls.Log)
        make_dir(cls.ComicProcess)
        make_dir(cls.MangaProcess)
        make_dir(cls.ManhuaProcess)
        make_dir(cls.MomoYolo)
        make_dir(cls.Storage)