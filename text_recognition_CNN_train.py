import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 基本配置
data_dir = "./dataset_chars_CNN"  # 训练集路径
batch_size = 64
num_epochs = 30
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file = "training_log.txt"
checkpoint_dir = "./checkpoints"

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # 模型、损失、优化器具体定义
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 清空原本日志文件
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Epoch,Loss,Accuracy\n")

    # 进入训练模式
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Acc: {epoch_acc:.4f}")

        # 写入日志文件
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_loss:.4f},{epoch_acc:.4f}\n")

        # 保存每个 epoch 的模型
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"cnn_char_model_epoch{epoch}.pth"))

    print(f"Training complete. Logs saved to {log_file}, checkpoints saved in {checkpoint_dir}")
