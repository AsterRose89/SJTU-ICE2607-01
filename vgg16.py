import time
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader

print('Loading model: VGG16')
vgg_model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)

mean_values = [0.485, 0.456, 0.406]
std_values = [0.229, 0.224, 0.225]
normalization = transforms.Normalize(mean=mean_values, std=std_values)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalization
])

def vgg16_feature_extractor(input_data):
    feat = vgg_model.features(input_data)
    feat = vgg_model.avgpool(feat)
    return feat

def process_image_and_save_features(img_path, save_dir):
    print('Preparing image data...')
    img = default_loader(img_path)
    processed_img = preprocess(img)
    img_tensor = torch.unsqueeze(processed_img, dim=0)

    print('Extracting features...')
    start_time = time.time()
    img_feat = vgg16_feature_extractor(img_tensor)
    feat_np = img_feat.detach().cpu().numpy()
    process_time = time.time() - start_time
    print(f'Feature extraction completed in {process_time:.2f} seconds')

    print('Saving features...')
    np.save(save_dir, feat_np)