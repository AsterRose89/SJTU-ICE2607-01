import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


def pyramid(input_image, pyramid_levels, scale_ratio=0.5):
    pyramid_list = [input_image]
    current_frame = input_image
    # 遍历生成金字塔各层图像
    for level_idx in range(1, pyramid_levels):
        # 计算下采样后图像尺寸
        new_width_dim = int(current_frame.shape[1] * scale_ratio)
        new_height_dim = int(current_frame.shape[0] * scale_ratio)
        # Lanczos4插值方法
        current_frame = cv2.resize(
            src=current_frame,
            dsize=(new_width_dim, new_height_dim),
            interpolation=cv2.INTER_LANCZOS4
        )
        pyramid_list.append(current_frame)
    return pyramid_list

def suppression(response_map, win_size=3):
    suppressed_map = np.zeros_like(response_map)
    # 窗口遍历
    half_win = win_size // 2
    for row in range(half_win, response_map.shape[0] - half_win):
        for col in range(half_win, response_map.shape[1] - half_win):
            # 提取当前窗口区域
            local_window = response_map[
                           row - half_win: row + half_win + 1,
                           col - half_win: col + half_win + 1
                           ]
            # 判断当前点是否为窗口内最大值
            if response_map[row, col] == np.max(local_window):
                suppressed_map[row, col] = response_map[row, col]
    return suppressed_map


def harris_corner_detection(input_img, pyramid_levels=3, scale_ratio=0.5,
                                        block_dim=2, k_size=3, harris_k=0.1, threshold_ratio=0.01):
    # 调用金字塔构建函数
    img_pyramid = pyramid(input_img, pyramid_levels, scale_ratio)
    corner_keypoints = []
    # 遍历金字塔各层
    for level, img in enumerate(img_pyramid):
        # 转换灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray_img = np.float32(gray_img)
        # 计算Harris响应图
        harris_response = cv2.cornerHarris(
            src=gray_img,
            blockSize=block_dim,
            ksize=k_size,
            k=harris_k
        )
        # 计算角点阈值
        corner_threshold = threshold_ratio * harris_response.max()
        # 筛选满足阈值的角点坐标
        corner_coords = np.argwhere(harris_response > corner_threshold)
        # 转换角点坐标到原图像坐标系
        scale_reciprocal = (1 / scale_ratio) ** level
        for pt in corner_coords:
            keypoint = cv2.KeyPoint(
                x=float(pt[1] * scale_reciprocal),
                y=float(pt[0] * scale_reciprocal),
                size=1 * scale_reciprocal
            )
            corner_keypoints.append(keypoint)
    print("Finish multi-scale Harris corner detection")
    return corner_keypoints


def bilinear_interpolate(input_mat, x_coord, y_coord):
    # 确保坐标在图像范围内
    x1 = max(int(np.floor(x_coord)), 0)
    x1 = min(x1, input_mat.shape[1] - 1)
    y1 = max(int(np.floor(y_coord)), 0)
    y1 = min(y1, input_mat.shape[0] - 1)

    x2 = min(x1 + 1, input_mat.shape[1] - 1)
    y2 = min(y1 + 1, input_mat.shape[0] - 1)

    # 获取四个邻域点像素值
    pixel_q11, pixel_q21 = input_mat[y1, x1], input_mat[y1, x2]
    pixel_q12, pixel_q22 = input_mat[y2, x1], input_mat[y2, x2]

    # 计算插值权重与结果
    delta_x1, delta_y1 = x_coord - x1, y_coord - y1
    delta_x2, delta_y2 = x2 - x_coord, y2 - y_coord
    interpolated_val = (pixel_q11 * delta_x2 * delta_y2 +
                        pixel_q21 * delta_x1 * delta_y2 +
                        pixel_q12 * delta_x2 * delta_y1 +
                        pixel_q22 * delta_x1 * delta_y1)
    return interpolated_val


def sift_descriptors(input_image, keypoints_list):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if input_image.ndim == 3 else input_image
    gray_image = np.float32(gray_image)
    img_shape = gray_image.shape

    descriptors_list = []
    # 遍历每个关键点计算描述子，变量名调整
    for kp in keypoints_list:
        kp_x, kp_y = int(kp.pt[0]), int(kp.pt[1])

        # 提取16x16邻域窗口
        window_top = max(0, kp_y - 8)
        window_bottom = min(kp_y + 8, img_shape[0])
        window_left = max(0, kp_x - 8)
        window_right = min(kp_x + 8, img_shape[1])
        kp_window = gray_image[window_top:window_bottom, window_left:window_right]

        # 跳过尺寸不足16x16的窗口
        if kp_window.shape[0] < 16 or kp_window.shape[1] < 16:
            continue

        # 计算梯度幅度与方向
        grad_dx = cv2.Sobel(kp_window, cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_dy = cv2.Sobel(kp_window, cv2.CV_32F, dx=0, dy=1, ksize=3)
        grad_mag = np.sqrt(grad_dx ** 2 + grad_dy ** 2)
        grad_angle = np.rad2deg(np.arctan2(grad_dy, grad_dx)) % 360

        # 计算主方向
        orient_hist = np.zeros(36, dtype=np.float32)
        for i in range(grad_mag.shape[0]):
            for j in range(grad_mag.shape[1]):
                bin_index = int(grad_angle[i, j] // 10) % 36
                orient_hist[bin_index] += grad_mag[i, j]
        main_orient = np.argmax(orient_hist) * 10
        obj_orient_angle = grad_angle[8, 8] - main_orient

        # 构建128维SIFT描述子
        sift_desc = []
        for block_i in range(0, 16, 4):
            for block_j in range(0, 16, 4):
                block_hist = np.zeros(8, dtype=np.float32)
                for sub_i in range(4):
                    for sub_j in range(4):
                        # 物体坐标系到图像坐标系转换
                        obj_x = block_j + sub_j
                        obj_y = block_i + sub_i

                        img_x_coord = (obj_x * np.cos(np.deg2rad(obj_orient_angle)) -
                                       obj_y * np.sin(np.deg2rad(obj_orient_angle)))
                        img_y_coord = (obj_x * np.sin(np.deg2rad(obj_orient_angle)) +
                                       obj_y * np.cos(np.deg2rad(obj_orient_angle)))

                        # 双线性插值获取梯度值
                        interp_mag = bilinear_interpolate(grad_mag, img_x_coord, img_y_coord)
                        interp_angle = bilinear_interpolate(grad_angle, img_x_coord, img_y_coord)

                        # 角度归一化与直方图统计
                        interp_angle = (interp_angle + 360) % 360
                        hist_bin_idx = int(interp_angle // 45) % 8
                        block_hist[hist_bin_idx] += interp_mag
                sift_desc.extend(block_hist)

        # 描述子归一化，优化数值稳定性写法
        sift_desc = np.array(sift_desc, dtype=np.float32)
        sift_desc /= (np.linalg.norm(sift_desc) + 1e-6)
        descriptors_list.append(sift_desc)

    print("Finish SIFT descriptor computation")
    return np.array(descriptors_list)


def match_features(desc_set1, desc_set2, ratio_thresh=0.75):
    match_pairs = []
    # 标记已匹配的描述子
    is_matched = [False] * len(desc_set2)

    # 遍历第一组每个描述子
    for idx1, desc1 in enumerate(desc_set1):
        best_idx = None
        min_dist = float('inf')
        second_min_dist = float('inf')

        # 遍历第二组描述子寻找最佳匹配
        for idx2, desc2 in enumerate(desc_set2):
            if is_matched[idx2]:
                continue
            # 计算欧氏距离
            dist = np.linalg.norm(desc1 - desc2)
            # 更新最佳与次佳匹配
            if dist < min_dist:
                second_min_dist = min_dist
                min_dist = dist
                best_idx = idx2

        # 应用比率测试筛选匹配对
        if min_dist < ratio_thresh * second_min_dist:
            match_pairs.append((idx1, best_idx, min_dist))
            is_matched[best_idx] = True

    print("Finish feature matching")
    return match_pairs


def match_features_BFMatcher(desc1, desc2, ratio_thresh=0.75):
    # 初始化暴力匹配器
    bf_matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False
    )

    # KNN匹配（k=2）
    knn_matches = bf_matcher.knnMatch(desc1, desc2, k=2)

    # 比率测试筛选优质匹配
    good_match_list = []
    for match1, match2 in knn_matches:
        if match1.distance < ratio_thresh * match2.distance:
            good_match_list.append(match1)

    print("Finish feature matching")
    return good_match_list


def sift_detection(img_path, target_img_path, output_dir, ratio_thresh=0.75):

    # 读取图像
    query_img = cv2.imread(img_path)
    target_img = cv2.imread(target_img_path)
    if query_img is None or target_img is None:
        print("Error: 无法读取指定图像文件")
        return

    # 转换灰度图
    gray_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象与提取特征
    sift_detector = cv2.SIFT_create()
    kp_query, desc_query = sift_detector.detectAndCompute(gray_query, None)
    kp_target, desc_target = sift_detector.detectAndCompute(gray_target, None)

    # 暴力匹配与比率测试
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf_matcher.knnMatch(desc_query, desc_target, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    match_result = cv2.drawMatches(
        img1=query_img,
        keypoints1=kp_query,
        img2=target_img,
        keypoints2=kp_target,
        matches1to2=good_matches,
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 显示并保存结果
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_result, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Feature Matching")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'sift_matching.png'))
    plt.close()

    return good_matches, kp_query, kp_target


def main(ratio_thresh=0.75, pyramid_levels=3, scale_ratio=0.3,
         block_dim=2, k_size=3, harris_k=0.04, threshold_ratio=0.01):
    # 获取当前文件所在目录
    cwd = os.path.dirname(__file__)
    # 确保输出目录存在
    output_dir = os.path.join(cwd, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 目标图像路径
    target_img_path = os.path.join(cwd, 'img', 'target.jpg')
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        print(f"无法读取目标图像: {target_img_path}")
        return

    # 提取目标图像的特征点和描述子（自定义方法）
    target_keypoints = harris_corner_detection(
        target_img, pyramid_levels=pyramid_levels, scale_ratio=scale_ratio,
        block_dim=block_dim, k_size=k_size, harris_k=harris_k, threshold_ratio=threshold_ratio
    )
    target_descriptors = sift_descriptors(target_img, target_keypoints)

    best_match_image = None
    best_match_keypoints = None
    best_match_descriptors = None
    best_matches = []
    best_score = 0

    # 数据集图像所在目录
    dataset_dir = os.path.join(cwd, 'dataset')
    if not os.path.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return

    # 遍历数据集里的所有图片
    for img_name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_name)
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        search_img = cv2.imread(img_path)
        if search_img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 提取搜索图像的特征点和描述子（自定义方法）
        search_keypoints = harris_corner_detection(
            search_img, pyramid_levels=pyramid_levels, scale_ratio=scale_ratio,
            block_dim=block_dim, k_size=k_size, harris_k=harris_k, threshold_ratio=threshold_ratio
        )
        search_descriptors = sift_descriptors(search_img, search_keypoints)

        # 特征点匹配
        matches = match_features(target_descriptors, search_descriptors, ratio_thresh=ratio_thresh)

        # 以匹配数量作为分数
        score = len(matches)
        if score > best_score:
            best_score = score
            best_match_image = search_img
            best_match_keypoints = search_keypoints
            best_match_descriptors = search_descriptors
            best_matches = matches
        print(f"图像 {img_name} 的匹配点数: {score}")

    # 绘制最佳匹配结果（自定义方法）
    if best_match_image is not None and best_matches:
        # 转换为cv2.DMatch格式用于绘制
        cv2_matches = [cv2.DMatch(_queryIdx=m[1], _trainIdx=m[0], _imgIdx=0, _distance=m[2]) for m in best_matches]
        result_img = cv2.drawMatches(
            best_match_image, best_match_keypoints,
            target_img, target_keypoints,
            cv2_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 使用Matplotlib显示并保存图像
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Custom SIFT Best Match")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'custom_best_match.png'))
        plt.close()
    else:
        print("未找到足够好的匹配图像")

    # 与OpenCV自带SIFT对比（以数据集中第一张有效图像为例）
    dataset_images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if dataset_images:
        sample_img_path = os.path.join(dataset_dir, dataset_images[0])
        sift_detection(sample_img_path, target_img_path, output_dir, ratio_thresh=ratio_thresh)
    else:
        print("数据集中没有有效图像")


if __name__ == "__main__":
    main()