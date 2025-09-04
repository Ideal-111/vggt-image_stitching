import cv2
import numpy as np

def linearBlending(imgs, mask, output1):
       
    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype=np.uint8)
    img_right_mask = np.zeros((hr, wr), dtype=np.uint8)
    img_left_1, img_right_1 = imgs
       
    # 找到img_left 和 img_right 的mask部分(即非0部分) 
    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img_left[i, j]) > 0:
                img_left_mask[i, j] = 1
        
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img_right[i, j]) > 0:
                img_right_mask[i, j] = 1
        
    # 找到两图重合的部分
    overlap_mask = np.zeros((hr, wr), dtype=np.uint8)
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
        
       
    # 计算重叠区域的线性alpha值，即将色彩从 img_left 到 img_right 逐步过渡
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
    for i in range(hr): 
        minIdx = maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j
        
        if (minIdx == maxIdx): # 融合区域过小
            continue
            
        decrease_step = 1 / (maxIdx - minIdx)
        for j in range(minIdx, maxIdx + 1):
            alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
    img_left[mask] = img_right[mask]
    linearBlending_img = img_left
    cv2.imwrite(output1, linearBlending_img)
    # 线性混合
    for i in range(hr):
        for j in range(wr):
            if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                linearBlending_img[i, j] = alpha_mask[i, j] * img_left_1[i, j] + (1 - alpha_mask[i, j]) * img_right_1[i, j]
    
    return linearBlending_img