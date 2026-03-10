import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Tuple

# 核心配置参数
CONFIDENCE_PRIMARY = 0.5     # 主车牌阈值：只要超过0.5就算（保证不漏检）
CONFIDENCE_SECONDARY = 0.85  # 后续车牌阈值：必须极高置信度才算（防止误检）

TARGET_WIDTH = 300      # 透视校正后的目标宽度
TARGET_HEIGHT = 100     # 透视校正后的目标高度
PADDING_RATIO_V = 0.08  # 垂直扩张
PADDING_RATIO_H = 0.04  # 水平扩张

def get_line_normal_vector(p1: np.ndarray, p2: np.ndarray, length: float) -> np.ndarray:
    vector = p2 - p1
    normal = np.array([-vector[1], vector[0]])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(2)
    normal_unit = normal / norm
    return normal_unit * length

def expand_keypoints_by_ratio(kpts_pixels: np.ndarray, ratio_v: float, ratio_h: float) -> np.ndarray:
    L_top = np.linalg.norm(kpts_pixels[0] - kpts_pixels[1])
    L_bottom = np.linalg.norm(kpts_pixels[3] - kpts_pixels[2])
    L_left = np.linalg.norm(kpts_pixels[0] - kpts_pixels[3])
    L_right = np.linalg.norm(kpts_pixels[1] - kpts_pixels[2])
    
    W_avg = (L_top + L_bottom) / 2
    H_avg = (L_left + L_right) / 2
    
    Delta_V = H_avg * ratio_v
    Delta_H = W_avg * ratio_h
    new_kpts = kpts_pixels.copy()
    C = np.mean(kpts_pixels, axis=0)
    
    def get_signed_normal(p1, p2, p_test, length):
        vector = p2 - p1
        n1 = np.array([-vector[1], vector[0]])
        n2 = np.array([vector[1], -vector[0]])
        norm1 = np.linalg.norm(n1)
        if norm1 == 0: return np.zeros(2)
        n1_unit = n1 / norm1
        n2_unit = n2 / norm1
        if np.linalg.norm(p1 + n1_unit * length - C) > np.linalg.norm(p1 - C):
            return n1_unit * length
        else:
            return n2_unit * length
    
    N_top = get_signed_normal(kpts_pixels[0], kpts_pixels[1], C, Delta_V)
    N_right = get_signed_normal(kpts_pixels[1], kpts_pixels[2], C, Delta_H)
    N_bottom = get_signed_normal(kpts_pixels[2], kpts_pixels[3], C, Delta_V) 
    N_left = get_signed_normal(kpts_pixels[3], kpts_pixels[0], C, Delta_H)
    
    new_kpts[0] = kpts_pixels[0] + N_top + N_left
    new_kpts[1] = kpts_pixels[1] + N_top + N_right
    new_kpts[2] = kpts_pixels[2] + N_bottom + N_right
    new_kpts[3] = kpts_pixels[3] + N_bottom + N_left
    return new_kpts

def perspective_warp(img: np.ndarray, src_points: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    dst_points = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    return cv2.flip(warped, 1)

def detect_and_crop_plates(model_path: str, image_path: str, output_dir: str) -> Tuple[int, List[Image.Image], List[Image.Image]]:
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        return 0, [], []
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return 0, [], []

    detection_save_path = os.path.join(output_dir, 'detection_result')
    os.makedirs(detection_save_path, exist_ok=True)

    img_orig = cv2.imread(image_path)
    if img_orig is None:
        return 0, [], []
    img_h, img_w = img_orig.shape[:2]

    results = model(image_path, stream=False, imgsz=640)
    cropped_plates_original: List[Image.Image] = []
    cropped_plates_corrected: List[Image.Image] = []

    # 绘制并保存带框图 (labeled_original.jpg)
    if results:
        labeled_img = results[0].plot() 
        cv2.imwrite(os.path.join(detection_save_path, "labeled_original.jpg"), labeled_img)

    if results and results[0].boxes.data.numel() > 0:
        r = results[0]
        if r.keypoints is None or r.keypoints.xyn is None:
            return 0, [], []

        boxes_data = r.boxes.data.cpu().numpy()
        keypoints_normalized = r.keypoints.xyn.cpu().numpy() 

        # 按置信度降序排列
        sorted_indices = np.argsort(boxes_data[:, 4])[::-1]
        boxes_data = boxes_data[sorted_indices]
        keypoints_normalized = keypoints_normalized[sorted_indices]

        # 遍历所有检测结果
        valid_count = 0
        for i in range(len(boxes_data)):
            conf_score = boxes_data[i][4]
            
            # 核心优化：双重阈值过滤
            # 如果是第1张（最高置信度），使用主阈值（0.5）
            # 如果是第2张及以后，使用严格阈值（0.85），剔除似是而非的误检
            current_threshold = CONFIDENCE_PRIMARY if valid_count == 0 else CONFIDENCE_SECONDARY
            
            if conf_score < current_threshold:
                # print(f"DEBUG: 过滤掉低置信度结果 #{i}: {conf_score:.4f} < {current_threshold}")
                continue

            # 只有通过阈值的才处理
            current_kpts_norm = keypoints_normalized[i]
            kpts_x = current_kpts_norm[:, 0] * img_w
            kpts_y = current_kpts_norm[:, 1] * img_h
            kpts_pixels_original = np.column_stack((kpts_x, kpts_y)).astype(np.float32)

            x_min, y_min = kpts_pixels_original[:,0].min(), kpts_pixels_original[:,1].min()
            x_max, y_max = kpts_pixels_original[:,0].max(), kpts_pixels_original[:,1].max()
            orig_crop = img_orig[int(y_min):int(y_max), int(x_min):int(x_max)]
            if orig_crop.size == 0: continue
            
            orig_crop_rgb = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB)
            cropped_plates_original.append(Image.fromarray(orig_crop_rgb))

            kpts_pixels_expanded = expand_keypoints_by_ratio(kpts_pixels_original, PADDING_RATIO_V, PADDING_RATIO_H)
            kpts_pixels_expanded[:, 0] = np.clip(kpts_pixels_expanded[:, 0], 0, img_w - 1)
            kpts_pixels_expanded[:, 1] = np.clip(kpts_pixels_expanded[:, 1], 0, img_h - 1)
            
            corrected_img_bgr = perspective_warp(img_orig, kpts_pixels_expanded.astype(np.float32), TARGET_WIDTH, TARGET_HEIGHT)
            corrected_img_rgb = cv2.cvtColor(corrected_img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(corrected_img_rgb)
            cropped_plates_corrected.append(pil_img)
            
            pil_img.save(os.path.join(detection_save_path, f"plate_{valid_count}_conf_{conf_score:.2f}.jpg"))
            valid_count += 1

    num_plates = len(cropped_plates_original)
    with open(os.path.join(detection_save_path, f"{num_plates}.txt"), 'w') as f:
        f.write("")

    return num_plates, cropped_plates_original, cropped_plates_corrected

if __name__ == '__main__':
    # 简单测试用
    pass
