import cv2
import numpy as np
from PyQt6.QtCore import Qt
import core.torch as torch

from storage import GlobalStorage

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

def get_comictextdetector_mask(image):
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
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (input_tuple[1], input_tuple[0])
        ratio = input_tuple[1] / iw, input_tuple[0] / ih  # 宽度，高度比例

    dh, dw = int(dh), int(dw)

    if raw_shape[::-1] != new_unpad:  # 调整大小
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=Qt.black)  # 添加边框

    if to_tensor:
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.array([np.ascontiguousarray(image)]).astype(np.float32) / 255
        if to_tensor:
            image = torch.from_numpy(image).to(device)
            if half:
                image = image.half()

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(input_size, input_size))
    GlobalStorage.DNNNet.setInput(blob)
    blks, mask, lines_map = GlobalStorage.DNNNet.forward(GlobalStorage.UOLayerNames)
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
    mask = mask.astype(np.uint8)

    # map output to input img
    mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
    return mask