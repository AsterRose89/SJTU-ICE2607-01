import cv2
import numpy as np
import os
import pathlib
from glob import glob
from typing import Optional, List, Tuple
from pathlib import Path
from PIL import Image
import argparse


def protect_leftmost_char(img, candidate_x, crop_top=0, crop_bottom=None, sample_width=3, sample_ratio=0.6, margin_ratio=0.2, gray_thresh=20):
    """
    检查最左侧候选裁剪线是否会误切汉字，如果左右颜色相近则取消裁剪
    """
    h, w = img.shape[:2]
    if crop_bottom is None:
        crop_bottom = h

    # 限制候选线不超过图像
    candidate_x = max(0, min(candidate_x, w-1))

    # 取上下中间区域
    y_start = int(crop_top + margin_ratio * (crop_bottom - crop_top))
    y_end = int(crop_top + (margin_ratio + sample_ratio) * (crop_bottom - crop_top))
    y_start = max(crop_top, y_start)
    y_end = min(crop_bottom, y_end)

    # 左右采样矩形
    x_left_start = max(candidate_x - sample_width, 0)
    x_left_end   = candidate_x
    x_right_start = candidate_x
    x_right_end   = min(candidate_x + sample_width, w-1)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_patch = gray[y_start:y_end, x_left_start:x_left_end]
    right_patch = gray[y_start:y_end, x_right_start:x_right_end]

    if left_patch.size == 0 or right_patch.size == 0:
        return candidate_x  # 防止越界

    # 计算灰度均值
    left_mean = np.mean(left_patch)
    right_mean = np.mean(right_patch)

    # 如果均值差过小，认为是汉字，取消裁剪
    if abs(left_mean - right_mean) < gray_thresh:
        return 0  # 保留原图最左边
    else:
        return candidate_x  # 可以裁剪

def detect_long_lines(bin_img, orientation='h', edge_ratio=0.110, angle_tol=15, length_ratio=0.4):
    """
    使用梯度 + 边界响应检测每个边缘区域中置信度最高的一条线段
    """
    h, w = bin_img.shape
    if orientation == 'h':
        edge_h = int(h * edge_ratio)
        regions = [bin_img[:edge_h, :], bin_img[-edge_h:, :]]
        offsets = [0, h - edge_h]
    else:
        edge_w = int(w * edge_ratio)
        regions = [bin_img[:, :edge_w], bin_img[:, -edge_w:]]
        offsets = [0, w - edge_w]

    result_lines = []

    for region, offset in zip(regions, offsets):
        # 梯度增强
        gray = region if region.dtype==np.uint8 else (region*255).astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        if orientation == 'h':
            grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            proj = np.sum(np.abs(grad), axis=1)  # 每行梯度强度
            max_idx = np.argmax(proj)
            max_val = proj[max_idx]
            threshold = 0.3 * np.max(proj)
            detected = max_val >= threshold
            if detected:
                y = max_idx
                x1, x2 = 0, w-1
                y1, y2 = y, y
                y1 += offset
                y2 += offset
            else:
                # 默认边界线
                y = 0 if offset==0 else region.shape[0]-1
                x1, x2 = 0, w-1
                y1, y2 = y + offset, y + offset
        else:  # 'v'
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            proj = np.sum(np.abs(grad), axis=0)  # 每列梯度强度
            max_idx = np.argmax(proj)
            max_val = proj[max_idx]
            threshold = 0.3 * np.max(proj)
            detected = max_val >= threshold
            if detected:
                x = max_idx
                y1, y2 = 0, h-1
                x1, x2 = x, x
                x1 += offset
                x2 += offset
            else:
                # 默认边界线
                x = 0 if offset==0 else region.shape[1]-1
                y1, y2 = 0, h-1
                x1, x2 = x + offset, x + offset

        result_lines.append((x1, y1, x2, y2, detected))

    return result_lines


def visualize_detected_lines(img, output_path=None):
    """
    使用与分割/warp 一致的 detect_long_lines 结果绘制上/下/左/右边界线
    如果 output_path 为 None 则不写盘，仅返回可视化图（BGR numpy）
    """
    vis = img.copy()
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先得到二值化版本
    except Exception:
        gray = img.copy() if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 识别边缘线并绘制
    h_lines = detect_long_lines(bin_img, 'h')
    v_lines = detect_long_lines(bin_img, 'v')
    
    for (x1, y1, x2, y2, detected) in h_lines + v_lines:
        color = (0, 0, 255) if detected else (255, 0, 0)
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 当 output_path 不为 None 时写入文件
    if output_path is not None:
        op = pathlib.Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(op), vis)
        # print(f"Saved confident edge lines to {op}")
        
    return vis


def line_intersection(line1, line2):
    """
    计算两条线段的交点
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1*x1 + B1*y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2*x3 + B2*y3

    det = A1*B2 - A2*B1
    if det == 0:
        return ((x1+x2)//2, (y1+y2)//2)  # 平行或重合时，返回近似点
    
    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return (int(x), int(y))


def warp_plate_by_vertices(img, top, bottom, left, right, output_path=None):
    """
    裁剪矩形区域
    """
    # 根据四条边裁剪出矩形区域
    x_min = int(left[0])
    x_max = int(right[0])
    y_min = int(top[1])
    y_max = int(bottom[1])

    # 保证不越界
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img.shape[1]-1, x_max)
    y_max = min(img.shape[0]-1, y_max)

    # 裁剪矩形区域并写入文件
    warped = img[y_min:y_max, x_min:x_max].copy()

    if output_path is not None:
        cv2.imwrite(str(output_path), warped)
        # print(f"Warped plate saved to {output_path}")

    return warped


def crop_edges_by_lines(img):
    """
    裁剪可能的车牌边缘，并保护最左侧汉字
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    h, w = bin_img.shape
    top_crop = 0
    bottom_crop = h
    left_crop = 0
    right_crop = w
    
    # 上
    top_lines = detect_long_lines(bin_img, 'h', edge_ratio=0.110)
    if top_lines:
        max_y = max([y2 for _,_,_,y2,_ in top_lines])
        top_crop = max_y

    # 下
    bottom_lines = detect_long_lines(bin_img[::-1,:], 'h', edge_ratio=0.110)
    if bottom_lines:
        min_y = min([y2 for _,_,_,y2,_ in bottom_lines])
        bottom_crop = h - min_y

    # 左
    left_lines = detect_long_lines(bin_img, 'v', edge_ratio=0.110)
    if left_lines:
        candidate_left_x = left_lines[0][0]  # 原来的最左竖线
        # 保护最左汉字
        protected_left_x = protect_leftmost_char(img, candidate_left_x, crop_top=top_crop, crop_bottom=bottom_crop)
        left_crop = protected_left_x

    # 右
    right_lines = detect_long_lines(bin_img[:,::-1], 'v', edge_ratio=0.110)
    if right_lines:
        min_x = min([x2 for x1,_,x2,_,_ in right_lines])
        right_crop = w - min_x

    cropped = img[top_crop:bottom_crop, left_crop:right_crop]
    return cropped


def detect_edge_lines(bin_img, orientation='h', edge_ratio=0.110, angle_tol=15, length_ratio=0.8, max_lines_per_region=1):
    """
    检测每个边缘区域最长倾斜线
    """
    h, w = bin_img.shape
    lines_to_draw = []

    if orientation == 'h':
        edge_h = int(h * edge_ratio)
        regions = [bin_img[:edge_h, :], bin_img[-edge_h:, :]]
        offsets = [0, h - edge_h]
    else:
        edge_w = int(w * edge_ratio)
        regions = [bin_img[:, :edge_w], bin_img[:, -edge_w:]]
        offsets = [0, w - edge_w]

    for region, offset in zip(regions, offsets):
        edges = cv2.Canny(region, 50, 150)
        # HoughLinesP，允许倾斜
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=int(length_ratio * max(h, w)), maxLineGap=5)
        candidates = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if orientation == 'h' and abs(angle) <= angle_tol:
                    candidates.append((x1, y1 + offset, x2, y2 + offset))
                elif orientation == 'v' and abs(abs(angle) - 90) <= angle_tol:
                    candidates.append((x1 + offset, y1, x2 + offset, y2))
        # 只保留最边缘一条
        if candidates:
            if orientation == 'h':
                line = candidates[0] if offset == 0 else candidates[-1]  # 上区域：第一条，下区域：最后一条
            else:
                line = candidates[0] if offset == 0 else candidates[-1]  # 左区域：第一条，右区域：最后一条
            lines_to_draw.append(line)
        else:
            # 没有检测到线，用边界线代替
            if orientation == 'h':
                line = (0, offset, w, offset)
            else:
                line = (offset, 0, offset, h)
            lines_to_draw.append(line)

    return lines_to_draw


def crop_edges_by_sobel(img):
    """
    使用 Sobel 边缘检测结果裁剪图像边缘
    """
    top, bottom, left, right = detect_edge_lines(img)
    cropped = img[top:bottom, left:right]
    return cropped


def remove_small_components(bin_img: np.ndarray, min_area: int = 30) -> np.ndarray:
    """
    移除二值图像中面积小于 min_area 的连通域（白色连通域）
    """
    img = bin_img.copy()
    
    # 将二值图像规范到 0~255
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # 计算白色连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    out = np.zeros_like(img)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lab] = 255
    return out


def deskew_plate(bin_img: np.ndarray, orig_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用最小外接矩形对整块车牌做轻微 deskew（基于二值图）；若无法计算（空图），则返回原图
    """
    # 找外轮廓
    cnts, _ = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bin_img, orig_img
    # 合并所有轮廓为一个点集
    all_pts = np.vstack(cnts)
    rect = cv2.minAreaRect(all_pts)
    angle = rect[-1]
    if abs(angle) < 0.5:
        return bin_img, orig_img  # 无需校正
    (h, w) = bin_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_bin = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    deskewed_color = cv2.warpAffine(orig_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return deskewed_bin, deskewed_color


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    改进版预处理：CLAHE -> 中值滤波/双边滤波 -> 自适应阈值（更小 blockSize）-> 形态学开+闭 -> 小连通域移除
    返回二值图（uint8，0/255）
    """
    # 灰度 + CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 去噪（中值滤波优先处理椒盐噪声）
    # kernel_size 根据图像大小自适应（奇数）
    h, w = enhanced.shape
    ks = 3 if min(h, w) < 200 else 5
    enhanced = cv2.medianBlur(enhanced, ks)

    # 自适应阈值
    blk = 15 if w > 200 else 11
    if blk % 2 == 0:
        blk += 1
    th1 = cv2.adaptiveThreshold(enhanced, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, blk, 2)

    # 结合 Otsu 做双阈值融合，保留更多字符细节
    _, th2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_or(th1, th2)

    # 形态学去噪：先开运算（去小白点/孔），再闭运算（填充字符缺口）
    # kernel 根据图像 height 自适应
    kernel_h = max(1, h // 60)
    kernel_w = max(1, w // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3,kernel_w), max(3,kernel_h)))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 进一步去除孤立小连通域（椒盐噪点）
    # 设置 min_area 与图片宽高有关（防止误删细小汉字）
    min_area = max(20, (w * h) // 4000)  # heuristic
    bin_img = remove_small_components(bin_img, min_area=min_area)

    # 6. 再做一次小的轻微膨胀（确保字符连通），且避免过度
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel2, iterations=1)

    return bin_img


def vertical_projection_split(plate_bw: np.ndarray, expected_chars: int,
                              output_path: Optional[pathlib.Path] = None,
                              gap_pos: float = 0.32,
                              gap_radius: float = 0.02,
                              plate_color: str = "蓝色") -> List[Tuple[int,int]]:  # <- 新增 plate_color 参数
    """
    返回 [(x_start, x_end), ...]，根据车牌颜色采用不同的分割策略。
    """
    h, w = plate_bw.shape
    splits = []

    # 绿色车牌（新能源车牌）特殊处理：左侧2字，右侧6字
    if plate_color == "绿":
        # print(f"DEBUG: 检测到绿色车牌，采用 2+6 分割策略")
        # 计算空隙位置（绿色车牌通常有更明显的分隔）
        gap_center = int(0.27 * w)  # 绿色车牌分隔线更靠左
        gap_left = max(0, int((0.27 - gap_radius*1.3) * w))
        gap_right = min(w, int((0.27 + gap_radius*1.3) * w))

        # 左侧 2 字
        left_split = [i * gap_left // 2 for i in range(1, 2+1)]
        last = 0
        for x in left_split:
            splits.append((last, x))
            last = x

        # 右侧 6 字
        right_split = [gap_right + i * (w - gap_right) // 6 for i in range(1, 6+1)]
        last = gap_right
        for x in right_split:
            splits.append((last, x))
            last = x

    else:
        # 原有逻辑：蓝色/黄色车牌，左侧2字，右侧5字
        # print(f"DEBUG: 检测到{plate_color}车牌，采用 2+5 分割策略")
        gap_center = int(gap_pos * w)
        gap_left = max(0, int((gap_pos - gap_radius) * w))
        gap_right = min(w, int((gap_pos + gap_radius) * w))

        # 左侧 2 字
        left_split = [i * gap_left // 2 for i in range(1, 2+1)]
        last = 0
        for x in left_split:
            splits.append((last, x))
            last = x

        # 右侧 5 字
        right_split = [gap_right + i * (w - gap_right) // 5 for i in range(1, 5+1)]
        last = gap_right
        for x in right_split:
            splits.append((last, x))
            last = x

    # 可视化
    if output_path:
        vis = cv2.cvtColor(plate_bw, cv2.COLOR_GRAY2BGR)
        # 绿色：普通分割线
        for x_start, x_end in splits:
            cv2.line(vis, (x_start, 0), (x_start, h-1), (0, 255, 0), 1)
            cv2.line(vis, (x_end, 0), (x_end, h-1), (0, 255, 0), 1)
        # 紫色：空隙中心及边界
        for x in [gap_left, gap_center, gap_right]:
            cv2.line(vis, (x, 0), (x, h-1), (255, 0, 255), 1)
        
        # 添加颜色信息到可视化图像
        color_text = f"Color: {plate_color}, Chars: {len(splits)}"
        cv2.putText(vis, color_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(str(output_path / "projection_split_vis.png"), vis)

    return splits


def save_char_images(bin_img, orig_img, splits, output_dir: Optional[pathlib.Path] = None) -> list:
    """
    保存字符图片，并在原图上绘制分割框 (plate_vis.png)，并返回 chars 列表（每项 28x28 numpy 图像）
    """
    # 内收字符框，从而得到更小更精确的字符
    INSET_RATIO = 0.06            # 水平内收比例
    VERTICAL_INSET_RATIO = 0.10   # 垂直内收比例，设置更小，减少上下车牌边缘线的干扰

    # 兼容 str 路径输入
    if output_dir is not None and not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    chars = []
    char_save_path = None
    vis_output_path = None

    # 保存字符输出与分割可视化输出
    if output_dir is not None:
        char_save_path = output_dir / "chars"
        char_save_path.mkdir(parents=True, exist_ok=True)
        vis_output_path = output_dir / "plate_vis.png"

    for idx, (x_start, x_end) in enumerate(splits):
        x_start = max(0, int(x_start))
        x_end = min(orig_img.shape[1], int(x_end))
        if x_end <= x_start:
            continue

        # 水平和垂直向内收缩（最小化边框/噪声）
        width = x_end - x_start
        dx = int(round(width * INSET_RATIO / 2.0))
        x0 = x_start + dx
        x1 = x_end - dx
        # 保证不越界且宽度合理
        if x1 <= x0 + 2:
            x0 = x_start
            x1 = x_end

        # 垂直方向按比例裁掉上下小部分
        h = orig_img.shape[0]
        vy = int(round(h * VERTICAL_INSET_RATIO / 2.0))
        y0 = max(0, vy)
        y1 = max(1, h - vy)
        if y1 <= y0 + 2:
            y0 = 0
            y1 = h

        # 使用裁剪后的坐标提取
        roi = bin_img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # 转化为 28x28 的格式
        try:
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except Exception:
            roi_resized = cv2.resize(cv2.copyMakeBorder(roi,0,0,0,0,cv2.BORDER_CONSTANT, value=0), (28,28))

        chars.append(roi_resized)

        if char_save_path:
            char_path = char_save_path / f"char_{idx:02d}.png"
            cv2.imwrite(str(char_path), roi_resized)

    # 在原图上画分割框并保存以可视化
    if vis_output_path is not None:
        vis = orig_img.copy()
        for x_start, x_end in splits:
            xs = max(0, int(x_start))
            xe = min(vis.shape[1], int(x_end))
            cv2.rectangle(vis, (xs, 0), (xe, vis.shape[0]), (0, 255, 0), 1)
        cv2.imwrite(str(vis_output_path), vis)

    return chars


def batch_process(input_dir, output_dir, expected_chars=7):
    """
    批量处理多张车牌图像：
    1. 可视化边缘检测
    2. 计算透视变换
    3. 保存 warped、分割可视化、chars 图片
    """
    img_paths = sorted(glob(os.path.join(input_dir, "*.*")))
    print(f"Found {len(img_paths)} images in {input_dir}")

    # 确保输出目录存在（外层）
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        filename = Path(img_path).stem
        print(f"Processing {filename} ...")

        # 使用 Path 创建当前图片目录
        plate_output_dir = output_dir / filename
        plate_output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping.")
            continue

        # 车牌边缘检测可视化
        vis_output = plate_output_dir / "edge_lines.png"
        edge_lines = visualize_detected_lines(img, vis_output)

        # 透视变换逻辑（裁剪之前）
        gray_bin = preprocess_plate(img)
        h_lines = detect_long_lines(gray_bin, 'h', edge_ratio=0.110)
        v_lines = detect_long_lines(gray_bin, 'v', edge_ratio=0.110)

        if len(h_lines) == 2 and len(v_lines) == 2:
            warp_output = plate_output_dir / "plate_warped.png"
            top_line    = h_lines[0][:4]
            bottom_line = h_lines[1][:4]
            left_line   = v_lines[0][:4]
            right_line  = v_lines[1][:4]
            warped_plate = warp_plate_by_vertices(
                img, top_line, bottom_line, left_line, right_line, warp_output
            )
        else:
            print(f"Warning: failed to detect four edges for {filename}, skipping warp.")
            warped_plate = img.copy()

        # 裁剪（采用透视后的图片）
        img_cropped = warped_plate.copy()

        # 颜色识别
        try:
            from recognition.plate_color_recognition import recognize_plate_color
            # 将 OpenCV 图像转换为 PIL 图像用于颜色识别
            pil_img = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            plate_color = recognize_plate_color(pil_img)
            # print(f"DEBUG: 识别到车牌颜色: {plate_color}")
        except Exception as e:
            print(f"Warning: 颜色识别失败: {e}")
            plate_color = "蓝"  # 默认值

        # 二值化处理
        bin_img = preprocess_plate(img_cropped)
        
        # 传入颜色参数进行字符分割（绿牌则为新能源，应有8个字符，否则有7个字符）
        splits = vertical_projection_split(
            bin_img,
            expected_chars=expected_chars,
            output_path=plate_output_dir,
            plate_color=plate_color
        )
       
        chars = save_char_images(bin_img, img_cropped, splits, plate_output_dir)

        print(f"✔ Results saved under: {plate_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--expect", type=int, default=7)
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_dir, expected_chars=args.expect)
