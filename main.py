import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import extract_feature
from vgg16 import vgg16_feature_extractor
from inception import get_inception_features


DATASET_DIR = "Dataset"
FEAT_DIRS = {
    "ResNet50": "Feature_resnet50",
    "VGG16": "Feature_vgg16",
    "InceptionV3": "Feature_inceptionv3"
}
QUERY_IMAGE_DIR = "Query_image"
QUERY_FEAT_DIR = "Query_feature"
OUTPUT_DIR = "Output"


def compute_angle_similarity(feat_a, feat_b):

    vec_a = feat_a.flatten()
    vec_b = feat_b.flatten()
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    angle_sim = np.arccos(cos_sim) / np.pi * 180
    return angle_sim


def batch_extract_features(model_extractor, data_dir, save_dir, img_count=50):

    for img_idx in range(1, img_count + 1):
        img_path = os.path.join(data_dir, f"{img_idx}.jpg")
        feat_save_path = os.path.join(save_dir, f"{img_idx}.npy")
        model_extractor(img_path, feat_save_path)


def batch_extract_resnet50(data_dir, save_dir, img_count=50):
    for img_idx in range(1, img_count + 1):
        img_path = os.path.join(data_dir, f"{img_idx}.jpg")
        feat_save_path = os.path.join(save_dir, f"{img_idx}.npy")

        test_image = extract_feature.default_loader(img_path)
        input_image = extract_feature.trans(test_image)
        input_image = torch.unsqueeze(input_image, 0)  # 现在 torch 已导入，可正常调用
        image_feature = extract_feature.features(input_image)
        image_feature = image_feature.detach().numpy()
        np.save(feat_save_path, image_feature)

def batch_extract_vgg16(data_dir, save_dir, img_count=50):
    from vgg16 import process_image_and_save_features  # 导入正确的VGG16特征保存函数
    for img_idx in range(1, img_count + 1):
        img_path = os.path.join(data_dir, f"{img_idx}.jpg")
        feat_save_path = os.path.join(save_dir, f"{img_idx}.npy")
        process_image_and_save_features(img_path, feat_save_path)  # 调用带路径参数的函数

def batch_extract_inception(data_dir, save_dir, img_count=50):
    from inception import process_and_save_features  # 导入正确的inception特征保存函数
    for img_idx in range(1, img_count + 1):
        img_path = os.path.join(data_dir, f"{img_idx}.jpg")
        feat_save_path = os.path.join(save_dir, f"{img_idx}.npy")
        process_and_save_features(img_path, feat_save_path)  # 调用带路径参数的函数

def find_top_similar(query_feat_path, feat_dir, top_k=5):

    query_feat = np.load(query_feat_path)
    all_similarities = []
    all_indices = []

    for img_idx in range(1, 51):
        feat_path = os.path.join(feat_dir, f"{img_idx}.npy")
        try:
            dataset_feat = np.load(feat_path)
        except FileNotFoundError:
            print(f"Warning: Feature file for index {img_idx} not found")
            continue

        sim_score = compute_angle_similarity(query_feat, dataset_feat)
        all_similarities.append(sim_score)
        all_indices.append(img_idx)

    sim_array = np.array(all_similarities)
    idx_array = np.array(all_indices)
    sorted_idx = np.argsort(sim_array)[:top_k]

    return idx_array[sorted_idx], sim_array[sorted_idx]


def plot_similar_results(query_img_path, top_indices, top_similarities, model_name, save_path):

    fig, axes = plt.subplots(1, 6, figsize=(28, 5))
    fig.suptitle(model_name, fontsize=16)

    query_img = plt.imread(query_img_path)
    axes[0].imshow(query_img)
    axes[0].axis('off')
    axes[0].set_title('Query Image')

    for idx, (img_idx, sim_score) in enumerate(zip(top_indices, top_similarities)):
        img_path = os.path.join('./Dataset', f"{img_idx}.jpg")
        similar_img = plt.imread(img_path)
        axes[idx + 1].imshow(similar_img)
        axes[idx + 1].axis('off')
        axes[idx + 1].set_title(f'Top {idx + 1}\nIndex: {img_idx}\nSimilarity: {sim_score:.2f}°')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    DATASET_DIR = './Dataset'
    OUTPUT_DIR = './Output'
    FEAT_DIRS = {
        'ResNet50': './Feature_resnet50',
        'VGG16': './Feature_vgg16',
        'InceptionV3': './Feature_inception'
    }
    QUERY_FEAT_DIR = './Query_feature'
    QUERY_IMG_DIR = './Query_image'
    QUERY_IMG_COUNT = 3

    os.makedirs(QUERY_FEAT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dir_path in FEAT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    for query_idx in range(QUERY_IMG_COUNT):

        query_img_path = os.path.join(QUERY_IMG_DIR, f"{query_idx + 1}.jpg")

        if not os.path.exists(query_img_path):
            print(f"错误：待检索图像文件不存在 - {query_img_path}")
            continue
        print(f"\n{'=' * 50}")
        print(f"Processing query image: {query_img_path}")
        print(f"{'=' * 50}")

    print("Starting batch feature extraction...")
    batch_extract_resnet50(DATASET_DIR, FEAT_DIRS['ResNet50'])
    batch_extract_vgg16(DATASET_DIR, FEAT_DIRS['VGG16'])
    batch_extract_inception(DATASET_DIR, FEAT_DIRS['InceptionV3'])
    print("Batch feature extraction completed!")

    for query_idx in range(QUERY_IMG_COUNT):
        query_img_path = os.path.join(QUERY_IMG_DIR, f"{query_idx + 1}.jpg")
        print(f"\n{'=' * 50}")
        print(f"Processing query image: {query_img_path}")
        print(f"{'=' * 50}")

        query_feat_paths = {
            'ResNet50': os.path.join(QUERY_FEAT_DIR, f'query_resnet50_{query_idx}.npy'),
            'VGG16': os.path.join(QUERY_FEAT_DIR, f'query_vgg16_{query_idx}.npy'),
            'InceptionV3': os.path.join(QUERY_FEAT_DIR, f'query_inception_{query_idx}.npy')
        }

        for model_name, feat_dir in FEAT_DIRS.items():
            print(f"\n--- Querying with {model_name} ---")
            if model_name == 'ResNet50':
                test_image = extract_feature.default_loader(query_img_path)
                input_image = extract_feature.trans(test_image)
                input_image = torch.unsqueeze(input_image, 0)
                image_feature = extract_feature.features(input_image)
                image_feature = image_feature.detach().numpy()
                np.save(query_feat_paths[model_name], image_feature)
            elif model_name == 'VGG16':
                from vgg16 import process_image_and_save_features
                process_image_and_save_features(query_img_path, query_feat_paths[model_name])
            elif model_name == 'InceptionV3':
                from inception import process_and_save_features
                process_and_save_features(query_img_path, query_feat_paths[model_name])

            top_indices, top_sims = find_top_similar(query_feat_paths[model_name], feat_dir, top_k=5)

            print(f"Top 5 similar images using {model_name}:")
            for idx, (img_idx, sim) in enumerate(zip(top_indices, top_sims)):
                print(f"  Top {idx + 1}: Index={img_idx}, Similarity={sim:.2f}°")

            result_save_path = os.path.join(OUTPUT_DIR, f'{model_name}_{query_idx}.png')
            plot_similar_results(query_img_path, top_indices, top_sims, model_name, result_save_path)
            print(f"Result plot saved to: {result_save_path}")


if __name__ == '__main__':
    main()