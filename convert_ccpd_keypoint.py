import os
import shutil
from typing import List, Tuple

# 数据集路径设置与图片大小设置
DATASET_PATH = "../database_process/test_set" 
OUTPUT_DIR_NAME = 'test_set_valid'
IMG_WIDTH = 720
IMG_HEIGHT = 1160

def parse_keypoints_and_bbox(filename_parts: List[str]) -> Tuple[list, list]:
    """
    解析CCPD文件名中的角点坐标，并计算满足 YOLO Keypoint 格式要求的边界框
    YOLO Keypoint 格式强制要求标签以 Bounding Box (bbox) 开始
    """
    # 获取角点坐标字符串，例如 '420,560_277,559_278,502_421,503'
    points_str = filename_parts[3].split('_')
    
    # 解析四个角点的原始 (x, y) 坐标
    raw_points = []
    for p_str in points_str[:4]: # 只取前四个坐标点
        try:
            x, y = map(int, p_str.split(','))
            raw_points.append((x, y))
        except ValueError:
            print(f"Warning: Failed to parse point string '{p_str}'. Skipping file.")
            return None, None
            
    if len(raw_points) < 4:
        print(f"Warning: Only found {len(raw_points)} keypoints. Skipping file.")
        return None, None

    # 关键点排序与提取
    keypoints_ordered = [
        raw_points[3],
        raw_points[2],
        raw_points[1],
        raw_points[0]
    ]
    
    # 计算最小外接矩形作为目标检测的依据
    all_x = [p[0] for p in raw_points]
    all_y = [p[1] for p in raw_points]
    
    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_x)
    y_max = max(all_y)

    # 转换为 YOLO 格式: [x_center, y_center, width, height]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    bbox_raw = [x_center, y_center, width, height]
    
    return keypoints_ordered, bbox_raw


def normalize_points(points: List[Tuple[int, int]], img_width: int, img_height: int) -> List[float]:
    """将原始坐标归一化并展平为列表 [x1, y1, v1, x2, y2, v2, ...]"""
    normalized_flat_points = []
    # 设置 Keypoint 可见性 (v=2 表示可见且带标签)
    VISIBILITY = 2 
    for x, y in points:
        norm_x = x / img_width
        norm_y = y / img_height
        normalized_flat_points.extend([norm_x, norm_y, VISIBILITY])
    return normalized_flat_points


def normalize_bbox(bbox_raw: List[float], img_width: int, img_height: int) -> List[float]:
    """归一化 Bounding Box 坐标"""
    x_center, y_center, width, height = bbox_raw
    norm_x_center = x_center / img_width
    norm_y_center = y_center / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return [norm_x_center, norm_y_center, norm_width, norm_height]


# 准备工作目
new_images_path = os.path.join(OUTPUT_DIR_NAME, 'images')
new_labels_path = os.path.join(OUTPUT_DIR_NAME, 'labels')

# 清理并创建新的文件夹结构
if os.path.exists(OUTPUT_DIR_NAME):
    shutil.rmtree(OUTPUT_DIR_NAME)
os.makedirs(new_images_path, exist_ok=True)
os.makedirs(new_labels_path, exist_ok=True)

# 获取所有图像文件
try:
    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]
except FileNotFoundError:
    print(f"错误: 找不到数据集路径 '{DATASET_PATH}'，请检查路径是否正确")
    exit()

print(f"找到 {len(image_files)} 张图像文件，开始处理...")

# 设置车牌类别ID为0
CLASS_ID = 0 
count = 0

# 遍历数据集中的每张图片
for img_file in image_files:
    # 构建图片完整路径
    src_img_path = os.path.join(DATASET_PATH, img_file)
    
    try:
        # 按 '-' 分割文件名
        parts = img_file.split('-')
        
        # 确保文件名至少有4个 '-' 分隔的部分，否则跳过
        if len(parts) < 4:
            continue
            
        # 解析四角点和边界框
        keypoints_ordered, bbox_raw = parse_keypoints_and_bbox(parts)
        if keypoints_ordered is None or bbox_raw is None:
            continue
            
        # 归一化 Bounding Box 和关键点坐标
        norm_bbox = normalize_bbox(bbox_raw, IMG_WIDTH, IMG_HEIGHT)
        norm_keypoints_flat = normalize_points(keypoints_ordered, IMG_WIDTH, IMG_HEIGHT)
        
        # 拼接 YOLO Keypoint 标签内容
        label_content_parts = [str(CLASS_ID)] 
        label_content_parts.extend([f"{x:.6f}" for x in norm_bbox])
        label_content_parts.extend([f"{x:.6f}" for x in norm_keypoints_flat])
        label_content = ' '.join(label_content_parts)
        
        # 写入标签文件 (.txt)
        label_file_name = img_file.replace('.jpg', '.txt')
        label_file_path = os.path.join(new_labels_path, label_file_name)
        with open(label_file_path, 'w') as f:
            f.write(label_content)

        # 复制图片到目标文件夹
        dst_img_path = os.path.join(new_images_path, img_file)
        shutil.copyfile(src_img_path, dst_img_path)
        
        count += 1
        
    except Exception as e:
        print(f"处理文件 {img_file} 时发生错误: {e}")
        continue
