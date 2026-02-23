import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict

def recognize_plate_color(plate_image: Image.Image) -> str:
    """
    识别车牌颜色
    支持颜色：黑、白、蓝、绿、黄
    """
    # 转换PIL图像为OpenCV格式
    plate_np = np.array(plate_image)
    if plate_np.ndim == 3 and plate_np.shape[2] == 3:
        plate_np = cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)
    
    # 转换为HSV颜色空间，更适合颜色识别
    hsv = cv2.cvtColor(plate_np, cv2.COLOR_BGR2HSV)
    
    # 定义各种颜色的HSV范围
    color_ranges = {
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 50, 50], [40, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50])
    }
    
    # 计算每种颜色的像素比例
    color_scores = {}
    total_pixels = plate_np.shape[0] * plate_np.shape[1]
    
    for color_name, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        
        # 创建颜色掩膜
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        # 计算该颜色像素的比例
        color_ratio = np.sum(mask > 0) / total_pixels
        color_scores[color_name] = color_ratio
    
    # 选择比例最高的颜色
    detected_color = max(color_scores.items(), key=lambda x: x[1])
    
    # 设置阈值，避免误识别
    if detected_color[1] < 0.1:  # 如果最大颜色比例小于10%，认为是识别失败
        return "未知"
    
    # 中文字典映射
    color_map = {
        'blue': '蓝',
        'green': '绿', 
        'yellow': '黄',
        'white': '白',
        'black': '黑'
    }
    
    return color_map.get(detected_color[0], "未知")


def recognize_plate_color_advanced(plate_image: Image.Image) -> Tuple[str, Dict]:
    """
    高级颜色识别，返回颜色和置信度信息
    """
    plate_np = np.array(plate_image)
    if plate_np.ndim == 3 and plate_np.shape[2] == 3:
        plate_np = cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)
    
    hsv = cv2.cvtColor(plate_np, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 50, 50], [40, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50])
    }
    
    color_scores = {}
    total_pixels = plate_np.shape[0] * plate_np.shape[1]
    
    for color_name, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        color_ratio = np.sum(mask > 0) / total_pixels
        color_scores[color_name] = color_ratio
    
    # 找到最佳匹配
    best_color = max(color_scores.items(), key=lambda x: x[1])
    
    color_map = {
        'blue': '蓝',
        'green': '绿',
        'yellow': '黄', 
        'white': '白',
        'black': '黑'
    }
    
    if best_color[1] < 0.1:
        return "未知", color_scores
    
    return color_map.get(best_color[0], "未知"), color_scores
