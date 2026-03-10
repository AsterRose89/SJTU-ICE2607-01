import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """
    保证输入是灰度图，如果是彩色图则转换，否则直接复制返回
    """
    if img is None or img.size == 0:
        return np.array([])
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def clahe_equalize(gray: np.ndarray) -> np.ndarray:
    """
    使用 CLAHE（对比度受限自适应直方图均衡）增强局部对比度
    """
    if gray.size == 0:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)


def binarize_for_analysis(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    二值化：输出白底黑字(Black Character=0, White Background=255)，用于分割分析
    """
    if gray.size == 0:
        return gray
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else: 
        # 默认使用自适应阈值，避免光照不均问题
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 33, 2)
    return binary


def find_best_split_line(binary_img: np.ndarray, 
                        start_x: int, 
                        end_x: int,
                        ideal_x: int) -> int:
    """
    在 [start_x, end_x] 范围内，寻找垂直方向上的颜色变化次数最少的列
    Args:
        binary_img (np.ndarray): 白底黑字图像 (字符=0, 背景=255)
        start_x, end_x (int): 搜索的横向范围
        ideal_x (int): 理想分割线的X坐标（在范围无效时使用）
    Returns:
        int: 最佳分割线的 X 坐标
    """
    H, W = binary_img.shape
    
    # 确保裁剪范围有效
    start_x = max(0, start_x)
    end_x = min(W, end_x)
    
    if end_x <= start_x:
        return ideal_x
        
    roi = binary_img[:, start_x: end_x]
    
    # 计算垂直方向上的颜色变化次数，其中 vertical_edge_count 值越小，该列越“颜色单一”，是更好的分割线
    vertical_edge_count = np.sum(np.abs(np.diff(roi, axis=0)), axis=0) / 255
    
    # 找到局部范围内的绝对最小值索引
    relative_index = np.argmin(vertical_edge_count)
    
    # 转换回原始图像的 X 坐标
    return start_x + relative_index


def segment_characters_from_image(
    plate_img: np.ndarray,
    expected_char_count: int = 7,
    debug: bool = False,
    save_dir: Optional[str] = None,
    target_size: Tuple[int, int] = (28, 28)
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    使用“等距引导 + 最小垂直边缘计数搜索”方法进行字符分割，并增加回退插值机制
    """
    if plate_img is None or plate_img.size == 0:
        print("Error: plate_img is None or empty.")
        return [], []
    
    # 预处理：转灰度，增强对比度，二值化
    gray = ensure_gray(plate_img)
    enhanced = clahe_equalize(gray)
    binary_img = binarize_for_analysis(enhanced, method="adaptive") 
    
    H, W = binary_img.shape
    
    # 核心分割逻辑：顺序搜索 + 记录回退点
    split_lines = [0]
    avg_char_width = W / expected_char_count

    initial_offset = int(W / 20)  # 初始偏移量：W/20
    search_range_half = int(W / 16)  # 搜索窗口半宽：总窗口 W/8，则半宽 W/16
    fallback_indices = []  # 记录哪些分割线是通过等距引导回退的 (索引从 1 到 N-1)
    
    for i in range(1, expected_char_count):
        # 理想分割线位置 (粗定位)
        ideal_split_x = int(i * avg_char_width)
        if i == 1:
            ideal_split_x += initial_offset
        
        # 搜索范围 [start, end]
        search_start = max(split_lines[-1] + 1, ideal_split_x - search_range_half)
        search_end = min(W, ideal_split_x + search_range_half)
        
        best_split_x = ideal_split_x  # 默认回退值
        
        if search_end <= search_start:
             # 如果搜索范围无效，使用粗定位作为回退值，并标记
             # 此时必须使用 ideal_split_x 作为临时分割线，因为它在后续计算 i+1 线的 search_start 时需要
             best_split_x = ideal_split_x 
             fallback_indices.append(i)
        else:
            # 寻找最佳分割线：使用最小垂直边缘计数搜索
            best_split_x = find_best_split_line(
                binary_img, 
                search_start, 
                search_end,
                ideal_split_x
            )
            
        split_lines.append(best_split_x)
        
    split_lines.append(W)
    
    # 后处理：对所有回退线进行插值修正
    if fallback_indices:
        print(f"Warning: Found fallback split lines at indices {fallback_indices}. Applying interpolation.")
        
        i = 0
        while i < len(fallback_indices):
            start_i = fallback_indices[i]
            
            # 确定锚点 A_prev_index/Pos
            # A_prev 是回退区域左侧第一个非回退点 (索引 start_i - 1)
            A_prev_index = start_i - 1
            A_prev_pos = split_lines[A_prev_index]
            
            # 确定锚点 A_next_index/Pos
            # A_next 是回退区域右侧第一个非回退点
            end_i = start_i
            # 找到连续回退序列的终点 (end_i 仍是回退点)
            while i < len(fallback_indices) and fallback_indices[i] == end_i:
                i += 1
                end_i += 1
            
            # end_i 此时是回退序列后的第一个非回退线的索引（如果它小于等于 expected_char_count）
            # A_next 的索引是 end_i
            A_next_index = end_i
            A_next_pos = split_lines[A_next_index]
            
            # 计算需要均匀划分的段数 (N_segments)
            N_segments = A_next_index - A_prev_index
            
            # 对 start_i 到 A_next_index-1 之间的所有回退分割线进行均匀插值
            segment_length = (A_next_pos - A_prev_pos) / N_segments
            
            for j in range(start_i, A_next_index):
                # j 是当前回退线的索引 (1 到 expected_char_count-1)
                # k 是它在 A_prev 和 A_next 之间的相对序号 (从 1 开始)
                k = j - A_prev_index 
                split_lines[j] = int(A_prev_pos + k * segment_length)
                
    # 提取、归一化字符图像
    final_boxes = []
    chars = []
    
    for i in range(expected_char_count):
        x_start = split_lines[i]
        x_end = split_lines[i+1]
        if x_end <= x_start:
            continue
            
        char_crop = binary_img[:, x_start:x_end]
        w = x_end - x_start
        final_boxes.append((x_start, 0, w, H))
        max_dim = max(w, H)
        pad = int(max_dim * 0.1) 
        canvas_size = max_dim + 2 * pad
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255 
        
        start_x_pad = (canvas_size - w) // 2
        start_y_pad = (canvas_size - H) // 2
        canvas[start_y_pad:start_y_pad+H, start_x_pad:start_x_pad+w] = char_crop
        char_normalized = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)
        _, char_normalized = cv2.threshold(char_normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        chars.append(char_normalized)

    # 调试可视化
    if debug or save_dir:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_binary_final.png"), binary_img)
            
            for i, ch in enumerate(chars):
                cv2.imwrite(os.path.join(save_dir, f"char_{i:02d}.png"), ch)
            
        # 绘制最终选定的分割线（红色为分割线，绿色为边界框）
        for x_split in split_lines[1:-1]:
            cv2.line(vis, (x_split, 0), (x_split, H), (0, 0, 255), 2)
        for (x, y, w, h) in final_boxes:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, "debug_split_overlay.png"), vis)
        if debug:
            try:
                cv2.imshow("char segmentation", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                pass

    return chars, final_boxes


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="车牌图像路径")
    parser.add_argument("--expect", type=int, default=7)
    parser.add_argument("--out", default="char_output")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--w", type=int, default=28)
    parser.add_argument("--h", type=int, default=28)
    args = parser.parse_args()

    plate_img = cv2.imread(args.image)
    if plate_img is None:
        print(f"Error: Could not read image {args.image}")
        return

    chars, boxes = segment_characters_from_image(
        plate_img, 
        expected_char_count=args.expect, 
        debug=args.debug, 
        save_dir=args.out,
        target_size=(args.w, args.h)
    )

    if len(chars) == args.expect:
        # print(f"Segmentation successful. Found {len(chars)} characters.")
        print(f"分割成功，识别到的字符数目为 {len(chars)}")
    else:
        print(f"分割失败，识别到的字符数目为 {len(chars)}，实际应该识别到的字符数目为 {args.expect}")


if __name__ == "__main__":
    main_cli()
