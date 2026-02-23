import numpy as np
import matplotlib.pyplot as plt

accuracy_values = [
    49.00, 49.33, 59.46, 62.40, 50.84, 46.37, 67.64, 71.78, 69.67, 68.13,
    75.05, 74.38, 74.13, 76.61, 78.68, 72.58, 80.79, 77.77, 76.53, 77.43,
    85.43, 85.37, 85.71, 85.64, 85.76, 85.93, 86.02, 85.83, 85.99, 86.21
]

epoch_numbers = list(range(1, len(accuracy_values) + 1))
plt.figure()
plt.plot(epoch_numbers, accuracy_values, linestyle='-', marker='')  # 显式指定线条样式
plt.xlabel('Training Epoch')
plt.ylabel('Accuracy on Test Set (%)')
plt.title('ResNet20 Test Accuracy Performance on CIFAR-10 Dataset')
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.savefig('./test_acc.png', bbox_inches='tight')  # 添加紧凑布局参数