import os
import sys
import argparse
import pathlib
import numpy as np
import cv2
from typing import Optional, List, Dict, Any
from PIL import Image

# 导入核心识别模块
from recognition.plate_color_recognition import recognize_plate_color, recognize_plate_color_advanced
from recognition.split_and_recognition import (
    preprocess_plate, 
    detect_long_lines, 
    warp_plate_by_vertices, 
    vertical_projection_split, 
    save_char_images
)

def _to_numpy_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def recognize_plate(image_path: str,
                    det_model_path: str,
                    rec_model_path: str,
                    output_dir: Optional[str] = None,
                    expected_char_count: int = 7,
                    save_artifacts: bool = False) -> List[Dict[str, Any]]:
    """
    执行检测 + 多车牌循环识别
    返回: 一个包含所有车牌识别结果的字典列表
    [
        {'text': '京Axxxxx', 'color': '蓝', 'crop_path': 'path/to/crop'},
        ...
    ]
    """
    from detection.detect_and_crop_plates import detect_and_crop_plates

    # 1. 路径设置
    if output_dir:
        output_path = pathlib.Path(output_dir)
        image_stem = pathlib.Path(image_path).stem
        output_path = output_path / image_stem
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # 2. 检测与裁剪
    # YOLO 默认返回是按置信度排序的，所以 plates_original[0] 就是置信度最高的
    num_plates, plates_original, plates_corrected = detect_and_crop_plates(
        model_path=det_model_path,
        image_path=image_path,
        output_dir=output_path
    )
    
    results = []

    # 如果没检测到车牌
    if not plates_original:
        return []

    # 3. 循环处理每一个检测到的车牌
    for i, (pil_orig, pil_corr) in enumerate(zip(plates_original, plates_corrected)):
        
        # 为每个车牌创建独立的子文件夹 (plate_0, plate_1...)
        plate_output_dir = output_path / f"plate_{i}" if output_path else None
        if plate_output_dir:
            plate_output_dir.mkdir(parents=True, exist_ok=True)

        current_result = {
            'id': i,
            'text': "识别失败",
            'color': "未知",
            'crop_path': ""
        }

        # 颜色识别
        try:
            plate_color = recognize_plate_color(pil_orig)
            current_result['color'] = plate_color
            if save_artifacts and plate_output_dir:
                _, color_scores = recognize_plate_color_advanced(pil_orig)
                with open(plate_output_dir / "color.txt", 'w', encoding='utf-8') as f:
                    f.write(str(color_scores))
        except:
            pass

        # 逐字符识别
        plate_np = _to_numpy_bgr(pil_corr) 
        rectified_np = plate_np.copy()
        
        try:
            # 预处理与微调矫正
            plate_bw = preprocess_plate(plate_np)
            h_lines = detect_long_lines(plate_bw, 'h')
            v_lines = detect_long_lines(plate_bw, 'v')
            
            if len(h_lines) == 2 and len(v_lines) == 2:
                rectified_np = warp_plate_by_vertices(
                    plate_np, h_lines[0][:4], h_lines[1][:4], v_lines[0][:4], v_lines[1][:4],
                    output_path=plate_output_dir / "plate_warped.png" if plate_output_dir else None
                )
            else:
                if plate_output_dir:
                    cv2.imwrite(str(plate_output_dir / "plate_warped.png"), plate_np)

            # 记录最终矫正图路径用于GUI显示
            if plate_output_dir:
                current_result['crop_path'] = str(plate_output_dir / "plate_warped.png")

            # 分割
            plate_bw_final = preprocess_plate(rectified_np)
            splits = vertical_projection_split(
                plate_bw_final, 
                expected_chars=expected_char_count,
                output_path=(plate_output_dir if save_artifacts else None),
                plate_color=current_result['color']
            )
            
            # 提取小图
            chars = save_char_images(
                plate_bw_final, rectified_np, splits, 
                output_dir=(plate_output_dir if save_artifacts else None)
            )
            
            # CNN 识别
            from recognition.single_char_recognition import recognize_chars
            if chars:
                chars_text, _ = recognize_chars(chars, model_path=rec_model_path)
                current_result['text'] = chars_text
            else:
                current_result['text'] = "分割失败"
                
        except Exception as e:
            print(f"Plate {i} processing error: {e}")
            current_result['text'] = "处理出错"

        results.append(current_result)

    return results

def _main_cli():
    # 简单测试用
    pass 

if __name__ == '__main__':
    # sys.path.append(os.path.dirname(__file__))
    _main_cli()
