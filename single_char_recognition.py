import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict

# 导入训练定义和字符映射
from recognition.CNN_train.text_recognition_CNN_train import SimpleCNN

# CCPD 字符集（与训练时一致）
CCPD_CHARSET = [
    # 数字+字母部分（34类，按训练集文件夹的 ASCII 排序）
    "1","10","11","12","13","14","15","16","17","18","19",
    "2","20","21","22","23","24","25","26","27","28","29",
    "3","30","31","32","33","34",
    "4","5","6","7","8","9",
    
    # 汉字部分 CN_1~CN_31（包含 CN_26 占位）
    "CN_1","CN_10","CN_11","CN_12","CN_13","CN_14","CN_15","CN_16","CN_17",
    "CN_18","CN_19","CN_2","CN_20","CN_21","CN_22","CN_23","CN_24","CN_25",
    "CN_27","CN_28","CN_29","CN_3","CN_30","CN_31","CN_4","CN_5","CN_6",
    "CN_7","CN_8","CN_9"
]

# 训练类名到字符值映射
class_to_char = {
    # 数字+字母部分
    "1":"A","10":"K","11":"L","12":"M","13":"N","14":"P","15":"Q","16":"R",
    "17":"S","18":"T","19":"U","2":"B","20":"V","21":"W","22":"X","23":"Y",
    "24":"Z","25":"0","26":"1","27":"2","28":"3","29":"4","3":"C","30":"5",
    "31":"6","32":"7","33":"8","34":"9","4":"D","5":"E","6":"F","7":"G","8":"H","9":"J",

    # 汉字部分 CN_1~CN_31
    "CN_1":"皖","CN_2":"沪","CN_3":"津","CN_4":"渝","CN_5":"冀",
    "CN_6":"晋","CN_7":"蒙","CN_8":"辽","CN_9":"吉","CN_10":"黑",
    "CN_11":"苏","CN_12":"浙","CN_13":"京","CN_14":"闽","CN_15":"赣",
    "CN_16":"鲁","CN_17":"豫","CN_18":"鄂","CN_19":"湘","CN_20":"粤",
    "CN_21":"桂","CN_22":"琼","CN_23":"川","CN_24":"贵","CN_25":"云",
    "CN_27":"藏","CN_28":"陕","CN_29":"甘","CN_30":"青","CN_31":"宁"
}

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 类型判断辅助函数
def is_chinese_char(char: str) -> bool:
    chinese_chars = [
        "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
        "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
        "桂","琼","川","贵","云","藏","陕","甘","青","宁"
    ]
    return char in chinese_chars


def is_non_chinese_char(char: str) -> bool:
    non_chinese_chars = [
        "A","B","C","D","E","F","G","H","J","K",
        "L","M","N","P","Q","R","S","T","U","V",
        "W","X","Y","Z","0","1","2","3","4","5",
        "6","7","8","9"
    ]
    return char in non_chinese_chars


def recognize_chars(chars: List[np.ndarray], model_path: str) -> Tuple[str, List[Dict]]:
    """
    使用训练好的 CNN 模型对单字符图像列表进行识别
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = SimpleCNN(num_classes=len(CCPD_CHARSET))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    chars_text = ""
    chars_detail = []

    for idx, char_np in enumerate(chars):
        # 转 PIL 图像
        pil_img = Image.fromarray(char_np.astype(np.uint8)).convert("L")
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            prob = F.softmax(out, dim=1)
            conf, idx_pred = prob.max(dim=1)
            idx_pred_val = int(idx_pred.item())
            conf_val = float(conf.item())
            key = CCPD_CHARSET[idx_pred_val]
            char = class_to_char.get(key, "")

        # 强制首位汉字
        if idx == 0:
            if not is_chinese_char(char):
                # 在汉字类别中选择置信度最高的
                chinese_indices = list(range(34, len(CCPD_CHARSET)))
                chinese_probs = prob[0, chinese_indices]
                max_conf, max_idx = chinese_probs.max(0)
                actual_idx = chinese_indices[max_idx.item()]
                key = CCPD_CHARSET[actual_idx]
                char = class_to_char.get(key, "")
                conf_val = float(max_conf.item())
        else:
            if not is_non_chinese_char(char):
                non_chinese_indices = list(range(0, 34))
                non_chinese_probs = prob[0, non_chinese_indices]
                max_conf, max_idx = non_chinese_probs.max(0)
                actual_idx = non_chinese_indices[max_idx.item()]
                key = CCPD_CHARSET[actual_idx]
                char = class_to_char.get(key, "")
                conf_val = float(max_conf.item())

        chars_text += char
        chars_detail.append({
            "index": idx,
            "char": char,
            "key": key,
            "confidence": conf_val
        })
    
    return chars_text, chars_detail
