import os
from pathlib import Path
import cv2
import numpy as np

from preprocess_utils import (
    preprocess_plate,
    vertical_projection_split_2,
    save_char_images,
    perspective_warp_plate
)

# CCPD 字符集
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙",
             "京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵",
             "云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","O"]  # 0~25
ADS = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z",
       "0","1","2","3","4","5","6","7","8","9","O"]


def parse_vertices_from_filename_ccpd(filename: str) -> np.ndarray:
    base = os.path.basename(filename)
    parts = base.split('-')
    vertex_field = parts[3]
    vertex_strs = vertex_field.split('_')
    pts = [list(map(int, s.split(','))) for s in vertex_strs]
    br, bl, tl, tr = pts[0], pts[1], pts[2], pts[3]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def parse_plate_indices(filename: str) -> list[int]:
    base = os.path.basename(filename)
    parts = base.split('-')
    char_field = parts[4]
    indices = list(map(int, char_field.split('_')))
    return indices


def get_char_folder_name(char_idx: int, pos: int) -> str:
    """
    pos: 0=省份, 1=字母, 2~=ads
    """
    if pos == 0:
        return f"CN_{char_idx+1}"  # 汉字索引从1开始
    else:
        return str(char_idx+1)  # 其他保持原有


if __name__ == "__main__":
    input_dir = Path("../database_process/CCPD-YOLO-100k")
    output_dir = Path("./dataset_chars")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_list = list(input_dir.glob("*.*"))
    if not img_list:
        raise FileNotFoundError(f"No images found in {input_dir}")

    for img_path in img_list:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: cannot read {img_path}, skipped")
            continue

        try:
            vertices = parse_vertices_from_filename_ccpd(img_path)
            char_indices = parse_plate_indices(img_path)
        except Exception as e:
            print(f"Failed parsing {img_path}: {e}")
            continue

        # 透视变换得到矩形车牌
        warped_plate = perspective_warp_plate(img, vertices, target_size=(240,80))
        bin_plate = preprocess_plate(warped_plate)

        # 字符分割，自动根据字符位数
        splits = vertical_projection_split_2(bin_plate, expected_chars=len(char_indices), output_path=None)
        chars = save_char_images(bin_plate, warped_plate, splits, output_dir=None)

        img_id = img_path.stem[-6:]
        num_chars_to_save = min(len(chars), len(char_indices))
        if num_chars_to_save < len(char_indices):
            print(f"Warning: {img_path.name} - expected {len(char_indices)} chars, but got {len(chars)} splits")

        for i in range(num_chars_to_save):
            idx = char_indices[i]
            folder_name = get_char_folder_name(idx, i)
            char_folder = output_dir / folder_name
            char_folder.mkdir(parents=True, exist_ok=True)
            char_filename = f"{img_id}_{i}.png"
            cv2.imwrite(str(char_folder / char_filename), chars[i])
