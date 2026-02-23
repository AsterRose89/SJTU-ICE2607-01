# License Plate Recognition System

A complete end-to-end License Plate Recognition (LPR) system integrating **YOLO-based plate detection**, **RetinaNet keypoint localization**, **perspective correction**, **HSV color classification**, **character segmentation**, and **dual recognition (CNN-based character classifier + RapidOCR text recognition)**.
A graphical user interface (GUI) is provided for easy operation, supporting **single-image recognition**, **batch processing**, and **multiple-plate detection**.

This repository contains the source code, model files, training utilities, and example results used in the system.


## Key Features

- **Accurate license plate localization**
  - yolov8n-pose for initial bounding-box detection
  - RetinaNet for precise 4-corner keypoint detection
- **Perspective correction** for skewed or angled plates
- **Robust plate color recognition** in HSV space (blue / green / yellow / white / black)
- **Character segmentation** using projection analysis & morphology
- **Two recognition pipelines**
  - CNN-based single-character classifier (ResNet-style backbone)
  - RapidOCR whole-plate sequence recognition
- **User-friendly GUI**
  - Multi-plate detection
  - Batch folder processing
  - Error handling (no plate / non-image / low-quality images)
- **Executable (.exe) build supported** via packaging
- **Complete training utilities** for detection, keypoint models, and recognition models


## Program Structure

```
LPR
│  build.bat                # Script for building the executable version
│  gui_app.py               # Main GUI application (primary entry point)
│  pipeline.py              # Core end-to-end pipeline logic
│  README.md
│  requirements.txt         # Project dependencies
│
├─ detection/
│   ├─ detect_and_crop_plates.py  # YOLO + RetinaNet hybrid detector
│   ├─ split.py                   # Utility for splitting plate regions
│   └─ database_procession/       # Detection model training workflow
│        ├─ best.pt               # YOLO/RetinaNet trained weights
│        ├─ convert_ccpd_keypoint.py
│        ├─ data.yaml
│        ├─ train.py
│        └─ license_plate_keypoint_project/
│             └─ train_ccpd_keypoint/
│                 ├─ args.yaml
│                 ├─ labels.jpg
│                 ├─ results.csv
│                 ├─ train_batch*.jpg
│                 └─ weights/     # Keypoint model final weights
│                     ├─ best.pt
│                     └─ last.pt
|
├─ dist/
│   └─ LPR_System
│        └─ LPR_System.exe     # Executable file for this project
│
├─ gui_results/      # Output folder for GUI-based recognition
│
├─ recognition/
│   ├─ best_recognition.pth
│   ├─ cnn_char_model.pth          # Character classifier
│   ├─ plate_color_recognition.py
│   ├─ plate_text_recognition.py
│   ├─ single_char_recognition.py
│   ├─ split_and_recognition.py
│   │
│   ├─ CNN_train/                  # Character recognition training scripts
│   │    ├─ build_ccpd_single_chars.py
│   │    ├─ preprocess_utils.py
│   │    ├─ text_recognition_CNN_train.py
│   │    └─ training_log.txt
│   │
│   └─ whole_text_recognition_train/
│        ├─ train_plate_recognition.py
│        └─ models_recognition/
│             ├─ best_recognition.pth
│             ├─ last_recognition.pth
│             └─ training_metrics.csv
│
└─ test/                           # Example test images
```

## System Pipeline

The system follows a modular workflow:

1. Plate Detection (yolov8n-pose + extra training)
2. Corner Keypoint Localization (RetinaNet)
3. Perspective Correction (cv2.getPerspectiveTransform)
4. Plate Color Recognition (HSV thresholds)
5. Character Segmentation (projection analysis + morphology)
6. Character Recognition
   - CNN classifier for segmented characters
   - RapidOCR for full-sequence recognition
7. Result visualization in GUI


## Installation

1. **Virtual Environment Creation and Dependency Installation**

```
conda create -n lpr python=3.10 -y
conda activate lpr
pip install -r requirements.txt
```

2. **Running the Application**

From the project root directory:

```
python ./gui_app.py
```

## GUI Preview

The GUI includes:
- Multi-plate detection with bounding boxes
- Real-time recognition results
- Table summarizing all processed images
- Error feedback when no plate is detected
- Exportable result folders under `gui_results/`


## Example Output Structure

For an input like `img1.jpg`(a valid photo input with one plate), the system produces:

```
gui_results/
 └─ img1/
      ├─ detection_result/
      │     labeled_original.jpg
      │     plate_0_conf_0.90.jpg
      │     1.txt
      │
      └─ plate_0/
           color.txt
           plate_vis.png
           plate_warped.png
           projection_split_vis.png
           chars/
               char_00.png
               char_01.png
               ...
```

This includes:

- Original image with bounding boxes
- Cropped plate image
- Warped (corrected) plate
- Character segmentation visualization
- Individual character crops
- Recognition text files


## Performance Summary

Based on experiments on **997 CCPD validation images**:

| Metric                     | Value            |
| -------------------------- | ---------------- |
| Localization Accuracy      | **98.39%**       |
| Recognition Accuracy       | **96.09%**       |
| Color Recognition Accuracy | **99.20%**       |
| Average Speed (CPU)        | **0.87 s/image** |
| Average Speed (GPU)        | **0.12 s/image** |
