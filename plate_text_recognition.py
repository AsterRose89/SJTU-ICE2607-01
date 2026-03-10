import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_recognition.pth")
# fallback char model (optional)
CHAR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "license_plate_model.pth")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.txt")

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','O']
ADS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','O']

CHAR_LIST = ['_']
for p in PROVINCES:
    if p != 'O':
        CHAR_LIST.append(p)
for a in ALPHABETS:
    if a != 'O':
        CHAR_LIST.append(a)
for d in ADS:
    if d != 'O':
        CHAR_LIST.append(d)

INDEX_TO_CHAR = {i: c for i, c in enumerate(CHAR_LIST)}
NUM_SEQ_CLASSES = len(CHAR_LIST)

# # 如果你确实不想回退，则保持 False
# ENABLE_FALLBACK = False

# 模型定义（与训练一致）
import torch.nn as nn
class FullPlateRecognitionNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        W = 120 // 8
        H_flat = (40 // 8) * 512
        self.W = W
        self.H_flat = H_flat
        self.sequence_head = nn.Sequential(
            nn.Linear(H_flat, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        features = features.permute(0, 3, 1, 2).contiguous()  # B, W', C, H'
        B, Wp, C, Hp = features.size()
        features = features.view(B * Wp, -1)
        logits = self.sequence_head(features)
        logits = logits.view(B, Wp, -1)
        output = logits.permute(1, 0, 2)  # T, B, C
        return F.log_softmax(output, dim=2)


# 加载模型（端到端）
def load_seq_model(seq_model_path=SEQ_MODEL_PATH):
    if not os.path.exists(seq_model_path):
        return None
    ckpt = torch.load(seq_model_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model = FullPlateRecognitionNet(NUM_SEQ_CLASSES)
    try:
        model.load_state_dict(state)
    except Exception:
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    return model


# 预处理（与训练一致）
def preprocess_for_seq(pil_img: Image.Image):
    img = pil_img.convert('L')
    img = img.resize((120, 40))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor  # [1,1,H,W]


# CTC 解码 & 置信度计算
def ctc_greedy_decode_and_confidence(log_probs):
    # log_probs: T, B, C (tensor)
    if log_probs.dim() != 3:
        raise ValueError("log_probs must be 3D tensor (T,B,C)")
    T, B, C = log_probs.size()
    # indices: T,B
    indices = torch.argmax(log_probs, dim=2)
    indices = indices.permute(1, 0)  # B, T
    blank_index = 0
    decoded_texts = []
    confidences = []
    # compute per-step max prob (softmax)
    with torch.no_grad():
        probs = log_probs.detach().exp()  # T,B,C -> probabilities
        max_probs, _ = probs.max(dim=2)  # T,B
        max_probs = max_probs.permute(1,0)  # B,T
    for b_idx, seq in enumerate(indices):
        last = blank_index
        chars = []
        step_probs = []
        for t_i, idx in enumerate(seq.cpu().tolist()):
            p_t = float(max_probs[b_idx, t_i].cpu().item())
            if idx != last and idx != blank_index:
                chars.append(INDEX_TO_CHAR.get(idx, '?'))
                step_probs.append(p_t)
            last = idx
        decoded = "".join(chars)
        # confidence: geometric mean of step_probs (if no char -> 0)
        if len(step_probs) == 0:
            conf = 0.0
        else:
            # avoid numerical underflow: use exp(mean(log))
            log_sum = sum(np.log(max(1e-12, p)) for p in step_probs)
            conf = float(np.exp(log_sum / len(step_probs)))
        decoded_texts.append(decoded)
        confidences.append(conf)
    return decoded_texts, confidences  # lists length B


# 内置 uniform split（仅用于可选回退）
def uniform_split_char_images(bgr_img, n_chars):
    """
    bgr_img: numpy HxW(x3)
    return: list of char images (grayscale numpy arrays) length n_chars
    """
    h, w = bgr_img.shape[:2]
    # crop central region slightly? For simplicity, use whole width
    # split width into n_chars equal parts
    widths = [w // n_chars] * n_chars
    # distribute remainder
    rem = w - sum(widths)
    for i in range(rem):
        widths[i] += 1
    xs = []
    x = 0
    for wi in widths:
        xs.append((x, x + wi))
        x += wi
    char_imgs = []
    for (l, r) in xs:
        crop = bgr_img[:, l:r]
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        # resize to classifier expected 28x28
        gray_rs = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_LINEAR)
        char_imgs.append(gray_rs)
    return char_imgs


_cached = {}
def get_models():
    if _cached:
        return _cached
    seq_m = load_seq_model()
    if seq_m is not None:
        _cached['mode'] = 'seq'
        _cached['seq_model'] = seq_m
        return _cached
    # # attempt to load char classifier if fallback enabled
    # if ENABLE_FALLBACK:
    #     try:
    #         # try to import your implementation if exists in recognition.recognition
    #         from recognition.recognition import load_recognizer, recognize_with_model
    #         char_tuple = load_recognizer(CHAR_MODEL_PATH, CLASSES_PATH)
    #         if char_tuple is not None:
    #             model_char, label_decoder, char_dev = char_tuple
    #             _cached['mode'] = 'char'
    #             _cached['char_model'] = model_char
    #             _cached['label_decoder'] = label_decoder
    #             _cached['char_device'] = char_dev
    #             return _cached
    #     except Exception:
    #         pass
    # _cached['mode'] = None
    # return _cached


def recognize_plate_whole(pil_img: Image.Image, enforce_lengths=None):
    """
    对车牌整体图像进行识别，并返回 meta 字典结构:
    {
        'len': int,              # 识别出的车牌长度
        'confidence': float,     # 置信度评分（CTC贪心解码）
        'is_green_pred': bool,   # 是否推断为绿牌（8位）
        'mode': 'seq' 或 'char' 或 None, 此处仅为seq处理，char版本放入另一个模块中
        'enforced': bool         # 若启用了 enforce_lengths，表示长度是否匹配
    }
    """
    models = get_models()
    mode = models.get('mode', None)
    
    # 模式 1：序列模型
    if mode == 'seq':
        model_seq = models['seq_model']
        inp = preprocess_for_seq(pil_img)  # [1,1,H,W]
        with torch.no_grad():
            logp = model_seq(inp)  # T,B,C
        decoded_list, confs = ctc_greedy_decode_and_confidence(logp)
        decoded = decoded_list[0] if decoded_list else ''
        conf = confs[0] if confs else 0.0
        L = len(decoded)
        is_green_pred = (L == 8)
        
        # enforce_lengths 检查
        if enforce_lengths is not None and L not in enforce_lengths:
            meta = {'len': L, 'confidence': conf, 'is_green_pred': is_green_pred, 'mode': 'seq', 'enforced': False}
        else:
            meta = {'len': L, 'confidence': conf, 'is_green_pred': is_green_pred, 'mode': 'seq', 'enforced': True}
        return decoded, meta

    # elif mode == 'char' and ENABLE_FALLBACK:
    #     # fallback: try uniform split for 7 and 8 (choose best by simple heuristic)
    #     arr = np.array(pil_img)
    #     if arr.ndim == 3 and arr.shape[2] == 3:
    #         bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    #     else:
    #         bgr = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    #     best_text = ''
    #     best_conf = -1.0
    #     best_len = 0
    #     # try both 7 and 8
    #     for n in (7, 8):
    #         chars_imgs = uniform_split_char_images(bgr, n)
    #         try:
    #             # assume recognize_with_model exists (you provided earlier)
    #             from recognition.recognition import load_recognizer, recognize_with_model
    #             # if not loaded earlier, load model now
    #             if 'char_model' not in _cached:
    #                 char_model, label_decoder, char_dev = load_recognizer(CHAR_MODEL_PATH, CLASSES_PATH)
    #                 _cached['char_model'] = char_model
    #                 _cached['label_decoder'] = label_decoder
    #                 _cached['char_device'] = char_dev
    #             text = recognize_with_model(chars_imgs, _cached['char_model'], _cached['label_decoder'], device=_cached['char_device'])
    #         except Exception:
    #             # if recognition function not available, fallback to blank
    #             text = ''
    #         # crude confidence: average max softmax of char model outputs (not computed here)
    #         conf = 0.0
    #         if text and len(text) == n:
    #             # prioritize longer (8) only if text non-empty
    #             if conf > best_conf:
    #                 best_conf = conf
    #                 best_text = text
    #                 best_len = n
    #     if best_text:
    #         meta = {'len': best_len, 'confidence': best_conf, 'is_green_pred': (best_len==8), 'mode': 'char'}
    #         return best_text, meta
    #     else:
    #         return '', {'len': 0, 'confidence': 0.0, 'is_green_pred': False, 'mode': None}

    else:
        # 没有可用模型：返回空结果
        return '', {'len': 0, 'confidence': 0.0, 'is_green_pred': False, 'mode': None}


if __name__ == '__main__':
    from PIL import Image
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plate_text_recognition.py /path/to/plate.jpg")
        sys.exit(1)
    p = sys.argv[1]
    img = Image.open(p)
    text, meta = recognize_plate_whole(img, enforce_lengths=(7,8))
    print("Decoded:", text)
    print("Meta:", meta)
