
import os
import hashlib
import re
import traceback

from loguru import logger

def printe(e):
    print(e)
    logger.error(e)
    traceback.print_exc()

def is_decimal_or_comma(s):
    pattern = r'^\d*\.?\d*$|^\d*[,]?\d*$'
    return bool(re.match(pattern, s))

# ================对文件算MD5================

def md5(path, blksize=2 ** 20):

    # 判断目标是否文件,及是否存在
    if os.path.isfile(path) and os.path.exists(path):  
        file_size = os.path.getsize(path)

        # if file_size <= 512MB
        if file_size <= 256 * 1024 * 1024:
            with open(path, 'rb') as f:
                cont = f.read()
            hash_object = hashlib.md5(cont)
            t_md5 = hash_object.hexdigest()
            return t_md5, file_size
        else:
            m = hashlib.md5()
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