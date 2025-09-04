import numpy as np

def transform_keypoints(kpts_original, original_size, target_size=518, mode="crop"):
    """
    根据图像预处理的逻辑转换关键点坐标。

    Args:
        kpts_original (np.ndarray): 原始图像上的关键点坐标，形状 (N, 2)。
        original_size (tuple): 原始图像的 (宽度, 高度)。
        target_size (int): 目标尺寸，默认为 518。
        mode (str): 预处理模式，"crop" 或 "pad"。

    Returns:
        np.ndarray: 转换后的关键点坐标，形状 (N, 2)。
    """
    if kpts_original.size == 0:
        return np.array([])

    original_width, original_height = original_size
    kpts_transformed = kpts_original.copy().astype(np.float32)

    if mode == "pad":
        # 模拟 load_and_preprocess_images 中的 pad 模式缩放
        if original_width >= original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height_unrounded = original_height * scale
            new_height = round(new_height_unrounded / 14) * 14
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width_unrounded = original_width * scale
            new_width = round(new_width_unrounded / 14) * 14
        
        # 应用缩放
        kpts_transformed *= scale

        # 模拟 load_and_preprocess_images 中的 pad 模式填充
        # 计算填充量
        h_padding = target_size - new_height
        w_padding = target_size - new_width
        
        pad_top = h_padding // 2
        pad_left = w_padding // 2

        # 加上填充偏移
        kpts_transformed[:, 0] += pad_left  # x 坐标
        kpts_transformed[:, 1] += pad_top   # y 坐标

    elif mode == "crop":
        # 模拟 load_and_preprocess_images 中的 crop 模式缩放
        scale = target_size / original_width
        new_width = target_size
        new_height_unrounded = original_height * scale
        new_height = round(new_height_unrounded / 14) * 14
        
        # 应用缩放
        kpts_transformed *= scale

        # 模拟 load_and_preprocess_images 中的 crop 模式中心裁剪
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            # 减去裁剪偏移
            kpts_transformed[:, 1] -= start_y

    return kpts_transformed

import cv2

def detectAndDescribe(img):
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(img, None)
    
    return kps, features