import os
import torch 
from ultralytics import YOLO

# 参数设置
PROJECT_NAME = 'license_plate_keypoint_project'
RUN_NAME = 'train_ccpd_keypoint'
MODEL_NAME = 'yolov8n-pose.pt'  # 使用 Pose 模型权重
DATA_YAML_PATH = 'data.yaml'
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640

# 自动计算断点文件 (last.pt) 的预期路径
CHECKPOINT_PATH = os.path.join(
    PROJECT_NAME, 
    RUN_NAME, 
    'weights', 
    'last.pt'
)


# 主训练模块
if __name__ == '__main__':
    # 检查数据集配置文件是否存在
    if not os.path.exists(DATA_YAML_PATH):
        print(f"错误：找不到配置文件 '{DATA_YAML_PATH}'")
        exit()
        
    # 检查训练集和验证集目录是否存在
    TRAIN_DATA_DIR = 'dataset_valid'
    VAL_DATA_DIR = 'test_set_valid'
    if not os.path.exists(TRAIN_DATA_DIR) or not os.path.exists(VAL_DATA_DIR):
        print(f"错误: 找不到数据集目录 '{TRAIN_DATA_DIR}' 或 '{VAL_DATA_DIR}'")
        exit()

    # 尝试断点重训和GPU训练
    try:
        if os.path.exists(CHECKPOINT_PATH):
            model = YOLO(CHECKPOINT_PATH)
        else:
            model = YOLO(MODEL_NAME) 

        if torch.cuda.is_available():
            device_to_use = '0' 
        else:
            device_to_use = 'cpu'
        
        # 打印训练参数
        print(f"模型训练参数:")
        print(f"   数据集配置: {DATA_YAML_PATH}")
        print(f"   初始模型: {model.ckpt_path}")
        print(f"   训练轮次 (Epochs): {EPOCHS}")
        
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=device_to_use,
            cache=True,
            project=PROJECT_NAME,
            name=RUN_NAME,
        )
        
        print(f"训练完成，最优模型权重已保存在 {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")

    except ImportError:
        print("错误: 缺少 ultralytics 库")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
