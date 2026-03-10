import os
import sys
from glob import glob
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import logging
import csv

# 基本配置
TRAIN_DATA_DIR = '../database_process/CCPD-YOLO-100k'
VAL_DATA_DIR = '../database_process/test_set'
SAVE_DIR = './models_recognition'
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_recognition.pth')
LAST_MODEL_PATH = os.path.join(SAVE_DIR, 'last_recognition.pth')
BAD_FILENAMES_LOG = os.path.join(SAVE_DIR, 'bad_filenames.log')
METRICS_CSV = os.path.join(SAVE_DIR, 'training_metrics.csv')

TARGET_IMG_SIZE = (120, 40)  # (W, H)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 1
DEBUG = False


# 字符集（稳定顺序）
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 构造稳定的字符列表（index 0 用作 CTC blank）
CHAR_LIST = []
CHAR_LIST.append('_')  # CTC Blank at index 0
for p in PROVINCES:
    if p != 'O':
        CHAR_LIST.append(p)
for a in ALPHABETS:
    if a != 'O':
        CHAR_LIST.append(a)
for d in ADS:
    if d != 'O':
        CHAR_LIST.append(d)

CHAR_TO_INDEX = {c: i for i, c in enumerate(CHAR_LIST)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(CHAR_LIST)  # 包含 blank


# 日志配置
os.makedirs(SAVE_DIR, exist_ok=True)
log_file = os.path.join(SAVE_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger("ccpd_recog")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
# console
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)
# file
fh = logging.FileHandler(log_file)
fh.setFormatter(fmt)
logger.addHandler(fh)


def decode_label_from_filename(filename: str):
    """
    从 CCPD 文件名解析四角顶点和字符索引，支持 7 或 8 个索引（返回 is_green_new 表示 8 字新能源）
    """
    try:
        base = os.path.basename(filename)
        parts = base.split('-')
        if len(parts) < 5:
            return None, None, None, None

        # 解析四顶点（parts[3]）
        vertex_part = parts[3].split('_')
        if len(vertex_part) != 4:
            _log_bad(filename, f"vertex_field_len={len(vertex_part)} != 4")
            return None, None, None, None
        vertices = []
        for v in vertex_part:
            try:
                x_str, y_str = v.split(',')
                x, y = int(x_str), int(y_str)
            except Exception:
                _log_bad(filename, f"vertex_parse_fail for '{v}'")
                return None, None, None, None
            vertices.append((x, y))
        vertices = np.array(vertices, dtype=np.float32).reshape(4, 2)

        # 解析第五字段：字符索引（parts[4]）
        idxs_part = parts[4].split('_')
        if len(idxs_part) not in (7, 8):
            _log_bad(filename, f"plate_index_len={len(idxs_part)} not in (7,8)")
            return vertices, None, None, None

        # 将索引字符串转为 ints
        try:
            indices = [int(i) for i in idxs_part]
        except Exception:
            _log_bad(filename, f"plate_index_parse_fail '{parts[4]}'")
            return vertices, None, None, None

        is_green_new = (len(indices) == 8)

        # 将 indices 转为字符（位置：0->province, 1->alphabet, >=2 -> ads）
        char_labels = []
        for i, idx in enumerate(indices):
            if i == 0:
                if idx < 0 or idx >= len(PROVINCES):
                    _log_bad(filename, f"province_index_out_of_range idx={idx}")
                    return vertices, None, None, None
                c = PROVINCES[idx]
            elif i == 1:
                if idx < 0 or idx >= len(ALPHABETS):
                    _log_bad(filename, f"alphabet_index_out_of_range idx={idx}")
                    return vertices, None, None, None
                c = ALPHABETS[idx]
            else:
                if idx < 0 or idx >= len(ADS):
                    _log_bad(filename, f"ads_index_out_of_range idx={idx} at pos {i}")
                    return vertices, None, None, None
                c = ADS[idx]
            # 如果存在占位符 'O'，当前我们视为无效（记录并丢弃）
            if c == 'O':
                _log_bad(filename, f"contains_placeholder_O at pos {i}")
                return vertices, None, None, None
            char_labels.append(c)

        # 将字符转为训练索引（使用 CHAR_TO_INDEX）
        try:
            ctc_indices = [CHAR_TO_INDEX[c] for c in char_labels]
        except KeyError as e:
            _log_bad(filename, f"char_to_index_missing: {e}")
            return vertices, None, None, None

        label_text = "".join(char_labels)
        return vertices, ctc_indices, label_text, is_green_new

    except Exception as e:
        # 解析失败
        _log_bad(filename, f"exception:{e}")
        return None, None, None, None


def _log_bad(filename: str, reason: str):
    """
    记录解析异常到单独文件（append）并打印到主 logger
    """
    msg = f"{os.path.basename(filename)}\t{reason}"
    try:
        with open(BAD_FILENAMES_LOG, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass
    logger.debug(f"Bad file: {msg}")


# 透视校正
def order_vertices(pts: np.ndarray) -> np.ndarray:
    """
    将任意 4 个点排序为 (tl, tr, br, bl) 并返回 float32 (4,2)
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.float32([tl, tr, br, bl])
    return ordered

def perspective_transform(img: np.ndarray, src_pts: np.ndarray, target_size: tuple) -> np.ndarray:
    W, H = target_size
    try:
        src_ordered = order_vertices(src_pts)
        dst_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        M = cv2.getPerspectiveTransform(src_ordered, dst_pts)
        rectified = cv2.warpPerspective(img, M, (W, H))
        if rectified.shape[1] != W or rectified.shape[0] != H:
            rectified = cv2.resize(rectified, (W, H))
        return rectified
    except Exception as e:
        # 返回中心裁剪或黑图
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        half_w, half_h = W // 2, H // 2
        cropped = img[max(0, cy - half_h):min(h, cy + half_h), max(0, cx - half_w):min(w, cx + half_w)]
        if cropped is None or cropped.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        return cv2.resize(cropped, (W, H))


# Dataset 与 collate 函数（返回 is_green_new）
class SequenceRecognitionDataset(Dataset):
    def __init__(self, data_dir, transform=None, verbose=True):
        self.data_dir = data_dir
        self.transform = transform
        image_paths = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
        self.samples = []
        if verbose:
            logger.info(f"Parsing dataset folder: {data_dir}, found {len(image_paths)} jpg files.")
        for path in tqdm(image_paths, desc=f"Parsing {os.path.basename(data_dir)}"):
            vertices, indices, label_text, is_green_new = decode_label_from_filename(path)
            if vertices is None or indices is None or label_text is None:
                # 过滤无效样本
                continue
            img = cv2.imread(path)
            if img is None:
                _log_bad(path, "imread_failed")
                continue
            self.samples.append({
                'path': path,
                'vertices': vertices,
                'indices': indices,
                'label_text': label_text,
                'is_green_new': is_green_new
            })
        if verbose:
            logger.info(f"Loaded {len(self.samples)} valid samples from {data_dir}.")
        if not self.samples:
            logger.warning(f"No valid samples in {data_dir}. Please check path and filename formats.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['path'])
        if img is None:
            img = np.zeros((TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[0], 3), dtype=np.uint8)
            rect = img
        else:
            rect = perspective_transform(img, s['vertices'], TARGET_IMG_SIZE)

        # 灰度单通道
        if len(rect.shape) == 3 and rect.shape[2] == 3:
            gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        else:
            gray = rect.copy()

        # 转为 tensor 并标准化
        if self.transform:
            img_tensor = self.transform(gray)
        else:
            img_tensor = transforms.ToTensor()(gray)

        label_tensor = torch.LongTensor(s['indices'])
        label_len = len(s['indices'])
        return img_tensor, label_tensor, label_len, s['path'], s['label_text'], s['vertices'], s['is_green_new']

def sequence_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, str, str, np.ndarray, bool]]):
    """
    batch 是 list of tuples: (img_tensor, label_tensor, label_len, path, label_text, vertices, is_green_new)
    返回分别打包的 tuple
    """
    imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    label_lens = [b[2] for b in batch]
    paths = [b[3] for b in batch]
    label_texts = [b[4] for b in batch]
    vertices = [b[5] for b in batch]
    is_green_news = [b[6] for b in batch]
    return imgs, labels, label_lens, paths, label_texts, vertices, is_green_news


# 模型 (简洁 LPRNet 风格)
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
        # 计算 W', H_flat
        self.W = TARGET_IMG_SIZE[0] // 8
        self.H_flat = (TARGET_IMG_SIZE[1] // 8) * 512

        self.sequence_head = nn.Sequential(
            nn.Linear(self.H_flat, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        # features shape: B, C, H', W'
        features = features.permute(0, 3, 1, 2).contiguous()  # B, W', C, H'
        B, Wp, C, Hp = features.size()
        features = features.view(B * Wp, -1)  # (B*W'), H_flat
        logits = self.sequence_head(features)  # (B*W', num_classes)
        logits = logits.view(B, Wp, -1)
        output = logits.permute(1, 0, 2)  # (T, B, C)
        return nn.functional.log_softmax(output, dim=2)


# CTC 解码（贪心）
def ctc_decode(log_probs: torch.Tensor):
    """
    输入 log_probs: shape (T, B, C) log-softmax 输出
    返回 decoded_texts: list length B
    """
    _, indices = torch.max(log_probs, dim=2)  # T, B
    indices = indices.permute(1, 0)  # B, T
    blank_index = CHAR_TO_INDEX['_']
    decoded_texts = []
    for seq in indices:
        last = blank_index
        chars = []
        for idx in seq.cpu().tolist():
            if idx != last and idx != blank_index:
                chars.append(INDEX_TO_CHAR.get(idx, '?'))
            last = idx
        decoded_texts.append("".join(chars))
    return decoded_texts


# 训练主函数
def train_sequence_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 注意输入为灰度图
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = SequenceRecognitionDataset(TRAIN_DATA_DIR, transform=transform, verbose=True)
    val_dataset = SequenceRecognitionDataset(VAL_DATA_DIR, transform=transform, verbose=True)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Error: Dataset empty. Please check paths and filename formats.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=sequence_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=sequence_collate_fn)

    model = FullPlateRecognitionNet(NUM_CLASSES).to(device)

    # 断言模型输出时间步长度 >= 最大期望标签长度（8）
    model_output_T = model.W
    max_expected_label_len = 8
    assert model_output_T >= max_expected_label_len, \
        f"Model output time steps T={model_output_T} < max expected label length {max_expected_label_len}. Increase TARGET_IMG_SIZE[0] or reduce stride."

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=CHAR_TO_INDEX['_'], reduction='mean', zero_infinity=True)

    start_epoch = 1
    best_acc = 0.0

    # 断点续训
    if os.path.exists(LAST_MODEL_PATH):
        try:
            ckpt = torch.load(LAST_MODEL_PATH, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 1) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            logger.info(f"Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # CSV metrics header
    csv_header = [
        'epoch', 'train_loss',
        'val_full_acc_overall', 'val_char_acc_overall',
        'val_full_acc_7', 'val_char_acc_7',
        'val_full_acc_8', 'val_char_acc_8'
    ]
    # 如果 CSV 文件不存在，写 header
    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(csv_header)

    logger.info(f"Starting training: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train")
        for imgs, labels, label_lens, paths, label_texts, vertices, is_green_news in train_pbar:
            imgs = torch.stack(imgs, dim=0).to(device)  # B, C, H, W
            labels_cat = torch.cat(labels).to(device)
            label_lens_tensor = torch.LongTensor(label_lens).to(device)

            log_probs = model(imgs)  # T, B, C
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long).to(device)

            loss = ctc_loss(log_probs, labels_cat, input_lengths, label_lens_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / num_samples if num_samples > 0 else 0.0

        # 验证阶段：计算 完整匹配率 + 字符级准确率（按类型统计）
        model.eval()
        correct_full_overall = 0
        total_overall = 0
        total_chars_overall = 0
        correct_chars_overall = 0

        # for 7-char
        correct_full_7 = 0
        total_7 = 0
        correct_chars_7 = 0
        total_chars_7 = 0

        # for 8-char
        correct_full_8 = 0
        total_8 = 0
        correct_chars_8 = 0
        total_chars_8 = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val")
            for imgs, labels, label_lens, paths, label_texts, vertices, is_green_news in val_pbar:
                imgs = torch.stack(imgs, dim=0).to(device)
                logp = model(imgs)
                preds = ctc_decode(logp)  # list len B

                # 目标文本（labels 是 list of tensors）
                target_texts = ["".join([INDEX_TO_CHAR[int(i.item())] for i in lbl]) for lbl in labels]

                for pred, tgt, is_g in zip(preds, target_texts, is_green_news):
                    total_overall += 1
                    if pred == tgt:
                        correct_full_overall += 1
                    # 字符级准确率（按最大长度作为分母）
                    minl = min(len(pred), len(tgt))
                    match_chars = sum(1 for i in range(minl) if pred[i] == tgt[i])
                    correct_chars_overall += match_chars
                    total_chars_overall += max(len(pred), len(tgt))

                    # 确认类型
                    if is_g:
                        total_8 += 1
                        if pred == tgt:
                            correct_full_8 += 1
                        minl8 = min(len(pred), len(tgt))
                        match_chars8 = sum(1 for i in range(minl8) if pred[i] == tgt[i])
                        correct_chars_8 += match_chars8
                        total_chars_8 += max(len(pred), len(tgt))
                    else:
                        total_7 += 1
                        if pred == tgt:
                            correct_full_7 += 1
                        minl7 = min(len(pred), len(tgt))
                        match_chars7 = sum(1 for i in range(minl7) if pred[i] == tgt[i])
                        correct_chars_7 += match_chars7
                        total_chars_7 += max(len(pred), len(tgt))

        val_acc_full_overall = correct_full_overall / total_overall if total_overall > 0 else 0.0
        val_char_acc_overall = correct_chars_overall / total_chars_overall if total_chars_overall > 0 else 0.0

        val_acc_full_7 = correct_full_7 / total_7 if total_7 > 0 else 0.0
        val_char_acc_7 = correct_chars_7 / total_chars_7 if total_chars_7 > 0 else 0.0

        val_acc_full_8 = correct_full_8 / total_8 if total_8 > 0 else 0.0
        val_char_acc_8 = correct_chars_8 / total_chars_8 if total_chars_8 > 0 else 0.0

        logger.info(
            f"Epoch {epoch} Summary: TrainLoss={avg_train_loss:.4f} | Val FullAcc Overall={val_acc_full_overall:.4f} | "
            f"CharAcc Overall={val_char_acc_overall:.4f} | 7-full={val_acc_full_7:.4f} | 7-char={val_char_acc_7:.4f} | "
            f"8-full={val_acc_full_8:.4f} | 8-char={val_char_acc_8:.4f}"
        )

        # 保存最新断点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        try:
            torch.save(checkpoint, LAST_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to save last checkpoint: {e}")

        # 保存最佳模型（以 overall full acc 判定）
        if val_acc_full_overall > best_acc:
            best_acc = val_acc_full_overall
            try:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                logger.info(f"New best model saved to {BEST_MODEL_PATH} with val_acc_full_overall={best_acc:.4f}")
            except Exception as e:
                logger.warning(f"Failed to save best model: {e}")

        # 写入 CSV
        row = [
            epoch, f"{avg_train_loss:.6f}",
            f"{val_acc_full_overall:.6f}", f"{val_char_acc_overall:.6f}",
            f"{val_acc_full_7:.6f}", f"{val_char_acc_7:.6f}",
            f"{val_acc_full_8:.6f}", f"{val_char_acc_8:.6f}"
        ]
        try:
            with open(METRICS_CSV, 'a', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Failed to write metrics csv: {e}")

    logger.info("Training finished.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    logger.info("=== CCPD Recognition Training Script (7/8 support) ===")
    logger.info(f"Train dir: {TRAIN_DATA_DIR}")
    logger.info(f"Val dir: {VAL_DATA_DIR}")
    logger.info(f"Save dir: {SAVE_DIR}")
    logger.info(f"CHAR LIST size (including blank): {NUM_CLASSES}")
    logger.info(f"Bad filenames log: {BAD_FILENAMES_LOG}")
    train_sequence_model()
