import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 图像路径
img_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

for img_path in img_paths:
    # 1. 读取彩色原图（OpenCV默认BGR格式）
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 2. 彩色图转灰度图（匹配实验报告的灰度转换公式）
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # 3. 构建灰度图保存路径（例：img1.jpg → img1_gray.jpg，与原图同目录）
    filename = os.path.basename(img_path)  # 获取文件名（如"img1.jpg"）
    name, ext = os.path.splitext(filename)  # 分离名称和后缀（name="img1", ext=".jpg"）
    gray_save_path = f"{name}_gray{ext}"  # 灰度图文件名（如"img1_gray.jpg"）
    # 4. 保存灰度图
    cv2.imwrite(gray_save_path, img_gray)

    
# 1. 彩色图像颜色直方图
for img_path in img_paths:
    # 以彩色方式读取图像
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # BGR 转 RGB（因为 matplotlib 显示的是 RGB 顺序）
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # 计算颜色直方图（分别计算 B、G、R 通道）
    hist_b = cv2.calcHist([img_color], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_color], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img_color], [2], None, [256], [0, 256])
    
    # 绘制颜色直方图
    plt.figure(figsize=(10, 6))
    plt.plot(hist_b, color='blue', label='Blue')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_r, color='red', label='Red')
    plt.title(f'Color Histogram of {img_path}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    # 保存颜色直方图图像
    plt.savefig(f'color_hist_{img_path.split(".")[0]}.png')
    plt.close()

# 2. 灰度图像灰度直方图和梯度直方图
for img_path in img_paths:
    # 以灰度方式读取图像
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 计算灰度直方图
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # 计算梯度（这里使用 Sobel 算子计算 x 和 y 方向梯度）
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # 将梯度幅值归一化到 0-255 范围
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 计算梯度直方图
    hist_grad = cv2.calcHist([grad_magnitude], [0], None, [256], [0, 256])
    
    # 绘制灰度直方图
    plt.figure(figsize=(10, 6))
    plt.plot(hist_gray, color='gray')
    plt.title(f'Gray Histogram of {img_path}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(f'gray_hist_{img_path.split(".")[0]}.png')
    plt.close()
    
    # 绘制梯度直方图
    plt.figure(figsize=(10, 6))
    plt.plot(hist_grad, color='black')
    plt.title(f'Gradient Histogram of {img_path}')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.savefig(f'grad_hist_{img_path.split(".")[0]}.png')
    plt.close()