import os
from shutil import copy2

import codecs
from csv import reader, writer

from utils import md5, printe

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
                    if md5(temp_csv) != md5(csv_path):
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
                if md5(temp_csv) != md5(csv_path):
                    copy2(temp_csv, csv_path)
                if temp_csv.exists():
                    os.remove(temp_csv)
            else:
                data_input.to_csv(csv_path, encoding='utf-8', index=False)
    except BaseException as e:
        printe(e)