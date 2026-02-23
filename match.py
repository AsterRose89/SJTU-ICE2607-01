import cv2
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class LocalitySensitiveHashing:
    def __init__(self, proj_matrices):
        self.proj_matrices = proj_matrices
        self.bucket_collections = [defaultdict(list) for _ in proj_matrices]
        self.img_filepaths = []

    def extract_color_feature(self, img):
        resized_img = cv2.resize(img, (200, 200))
        h, w = resized_img.shape[:2]
        half_h, half_w = h // 2, w // 2

        quadrants = [
            resized_img[:half_h, :half_w],
            resized_img[:half_h, half_w:],
            resized_img[half_h:, :half_w],
            resized_img[half_h:, half_w:]
        ]

        # 计算每个区域的颜色特征
        color_features = []
        for quad in quadrants:
            b_channel, g_channel, r_channel = cv2.split(quad)
            total = np.sum(b_channel) + np.sum(g_channel) + np.sum(r_channel)
            color_features.extend([
                np.sum(b_channel) / total,
                np.sum(g_channel) / total,
                np.sum(r_channel) / total
            ])
        return np.array(color_features)

    def build_image_index(self, img_directory):
        features_list = []

        # 遍历目录下所有JPG图像
        for img_file in Path(img_directory).glob('*'):
            if img_file.suffix.lower() == '.jpg':
                try:
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    # 提取特征并存储
                    feat = self.extract_color_feature(image)
                    self.img_filepaths.append(str(img_file))
                    features_list.append(feat)
                except Exception as e:
                    print(f"处理图像 {img_file} 时出错: {e}")

        # 构建哈希桶索引
        for feat_idx, feat_vec in enumerate(features_list):
            for bucket_idx, proj in enumerate(self.proj_matrices):
                hash_key = self.generate_hash_key(feat_vec, proj)
                self.bucket_collections[bucket_idx][hash_key].append((feat_idx, feat_vec))

    def generate_hash_key(self, feature_vec, projection):
        # 特征量化
        quantized = np.zeros(feature_vec.shape, dtype=int)
        quantized[feature_vec > 0.6] = 2
        quantized[(feature_vec >= 0.3) & (feature_vec <= 0.6)] = 1

        # 生成汉明码
        hamming_code = []
        for val in quantized:
            if val == 0:
                hamming_code += [0, 0]
            elif val == 1:
                hamming_code += [1, 0]
            elif val == 2:
                hamming_code += [1, 1]

        # 投影计算哈希值
        hamming_arr = np.array(hamming_code)
        projected_code = hamming_arr[projection]
        return tuple(projected_code)

    def search_similar_images(self, query_img_path, top_k=1):
        start = time.time()

        # 读取并处理查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("无法读取查询图像")
        query_feat = self.extract_color_feature(query_img)

        # 收集候选图像
        candidates = []
        visited = set()

        for bucket_idx, proj in enumerate(self.proj_matrices):
            hash_val = self.generate_hash_key(query_feat, proj)
            for idx, feat in self.bucket_collections[bucket_idx].get(hash_val, []):
                if idx not in visited:
                    candidates.append((idx, feat))
                    visited.add(idx)

        if not candidates:
            return [], time.time() - start

        # 计算距离并排序
        dist_list = []
        for idx, feat in candidates:
            squared_dist = np.sum((feat - query_feat) ** 2)
            dist_list.append((squared_dist, idx))

        # 按距离升序排序
        dist_list.sort()
        end = time.time()

        return [self.img_filepaths[idx] for _, idx in dist_list[:top_k]], end - start

    def display_matching_results(self, query_path, result_paths, save_file=None):
        # 读取图像
        query_img = cv2.imread(query_path)
        result_imgs = [cv2.imread(path) for path in result_paths]

        # 创建可视化窗口
        plt.figure(figsize=(6, 3))

        # 显示查询图像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title('查询图像')
        plt.axis('off')

        # 显示最佳匹配
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_imgs[0], cv2.COLOR_BGR2RGB))
        plt.title('最佳匹配')
        plt.axis('off')

        plt.suptitle('LSH匹配结果')
        plt.tight_layout()

        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()


class BruteForceMatcher:
    def __init__(self):
        self.image_paths = []
        self.feature_matrix = None

    def extract_color_histogram(self, image):
        # 统一图像尺寸
        img_resized = cv2.resize(image, (200, 200))
        h, w = img_resized.shape[:2]
        half_h, half_w = h // 2, w // 2

        # 分割为四个区域
        regions = [
            img_resized[:half_h, :half_w],
            img_resized[:half_h, half_w:],
            img_resized[half_h:, :half_w],
            img_resized[half_h:, half_w:]
        ]

        # 计算每个区域的颜色特征
        hist_features = []
        for reg in regions:
            b, g, r = cv2.split(reg)
            total_energy = np.sum(b) + np.sum(g) + np.sum(r)
            hist_features.extend([
                np.sum(b) / total_energy,
                np.sum(g) / total_energy,
                np.sum(r) / total_energy
            ])
        return np.array(hist_features)

    def create_image_index(self, img_dir):
        features = []
        # 遍历目录下的JPG图像
        for img_path in Path(img_dir).glob('*'):
            if img_path.suffix.lower() == '.jpg':
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    # 提取特征
                    feat = self.extract_color_histogram(img)
                    self.image_paths.append(str(img_path))
                    features.append(feat)
                except Exception as e:
                    print(f"处理 {img_path} 时发生错误: {e}")

        # 转换为numpy数组便于计算
        self.feature_matrix = np.array(features)

    def find_nearest_neighbors(self, query_img_path, k=1):
        start_time = time.time()

        # 读取并处理查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("无法读取查询图像")
        query_feat = self.extract_color_histogram(query_img)

        # 计算与所有图像的平方距离
        squared_distances = np.sum((self.feature_matrix - query_feat) ** 2, axis=1)

        # 获取距离最小的k个索引
        nearest_indices = np.argsort(squared_distances)[:k]

        end_time = time.time()
        search_duration = end_time - start_time

        return [self.image_paths[i] for i in nearest_indices], search_duration

    def plot_matching_results(self, query_path, result_paths, save_path=None):
        # 读取图像
        query_img = cv2.imread(query_path)
        result_imgs = [cv2.imread(path) for path in result_paths]

        # 创建画布
        plt.figure(figsize=(6, 3))

        # 显示查询图像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title('查询图像')
        plt.axis('off')

        # 显示最佳匹配
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_imgs[0], cv2.COLOR_BGR2RGB))
        plt.title('最佳匹配')
        plt.axis('off')

        plt.suptitle('NN结果')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()