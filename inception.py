import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

print('Loading model: InceptionV3')
inception_model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
normalizer = transforms.Normalize(mean=img_mean, std=img_std)

image_transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    normalizer
])

def get_inception_features(input_tensor):
    feat = inception_model.Conv2d_1a_3x3(input_tensor)
    feat = inception_model.Conv2d_2a_3x3(feat)
    feat = inception_model.Conv2d_2b_3x3(feat)
    feat = inception_model.Conv2d_3b_1x1(feat)
    feat = inception_model.Conv2d_4a_3x3(feat)
    feat = inception_model.Mixed_5b(feat)
    feat = inception_model.Mixed_5c(feat)
    feat = inception_model.Mixed_5d(feat)
    feat = inception_model.Mixed_6a(feat)
    feat = inception_model.Mixed_6b(feat)
    feat = inception_model.Mixed_6c(feat)
    feat = inception_model.Mixed_6d(feat)
    feat = inception_model.Mixed_6e(feat)
    feat = inception_model.Mixed_7a(feat)
    feat = inception_model.Mixed_7b(feat)
    feat = inception_model.Mixed_7c(feat)
    return feat


def process_and_save_features(img_path, output_path):
    print('Preparing image data...')
    img = default_loader(img_path)
    processed_img = image_transforms(img)
    img_tensor = torch.unsqueeze(processed_img, dim=0)

    print('Extracting features...')
    start_time = time.time()
    features = get_inception_features(img_tensor)
    features_np = features.detach().cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f'Feature extraction time: {elapsed_time:.2f} seconds')

    print('Saving features...')
    np.save(output_path, features_np)