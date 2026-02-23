import sys
import os
import shutil
from pathlib import Path
import time
import traceback

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGroupBox, QMessageBox, QProgressBar, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QScrollArea, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QColor, QBrush
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import pipeline

class BatchWorker(QThread):
    """
    后台工作线程：支持多车牌列表返回
    """
    # 信号变化：Results 变成了一个 list
    # (row, img_path, res_dir, results_list, t_cost)
    item_finished = pyqtSignal(int, str, str, object, float)
    all_finished = pyqtSignal()
    progress_update = pyqtSignal(int, int)

    def __init__(self, task_list, det_model, rec_model, base_out_dir):
        super().__init__()
        self.task_list = task_list 
        self.det_model = det_model
        self.rec_model = rec_model
        self.base_out_dir = base_out_dir
        self.is_running = True

    def run(self):
        total = len(self.task_list)
        for i, (table_row_idx, img_path) in enumerate(self.task_list):
            if not self.is_running: break
            
            t0 = time.time()
            try:
                img_stem = Path(img_path).stem
                target_dir = Path(self.base_out_dir) / img_stem
                
                if target_dir.exists():
                    try: shutil.rmtree(target_dir)
                    except: pass 

                # 调用 pipeline (现在返回的是 list)
                results_list = pipeline.recognize_plate(
                    image_path=img_path,
                    det_model_path=self.det_model,
                    rec_model_path=self.rec_model,
                    output_dir=self.base_out_dir,
                    expected_char_count=7,
                    save_artifacts=True
                )
                
                t_cost = time.time() - t0
                self.item_finished.emit(table_row_idx, img_path, str(target_dir), results_list, t_cost)

            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                t_cost = time.time() - t0
                # 异常时返回 None
                self.item_finished.emit(table_row_idx, img_path, "", None, t_cost)
            
            self.progress_update.emit(i + 1, total)

        self.all_finished.emit()

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SJTU ICE2607 - 车牌识别系统 (多目标检测版)")
        self.resize(1200, 850)

        self.det_model = "detection/database_procession/best.pt"
        self.rec_model = "recognition/cnn_char_model.pth"
        self.output_dir = "gui_results"
        
        self.table_data = {} 
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # === 1. 控制面板 ===
        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QHBoxLayout()
        
        self.btn_single = QPushButton("单张导入")
        self.btn_single.clicked.connect(self.load_single_image)
        self.btn_single.setMinimumHeight(40)

        self.btn_batch = QPushButton("批量导入")
        self.btn_batch.clicked.connect(self.load_batch_folder)
        self.btn_batch.setMinimumHeight(40)

        self.btn_run = QPushButton("开始识别")
        self.btn_run.clicked.connect(self.start_processing)
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setEnabled(False) 
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        ctrl_layout.addWidget(self.btn_single)
        ctrl_layout.addWidget(self.btn_batch)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.progress_bar)
        ctrl_layout.addWidget(self.btn_run)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # === 2. 核心展示区 ===
        img_layout = QHBoxLayout()
        
        # 左侧：原图
        self.group_orig = QGroupBox("检测结果")
        orig_vbox = QVBoxLayout()
        self.lbl_orig = QLabel("请在下方列表中选择图片")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig.setStyleSheet("background: #f0f0f0; border: 1px solid #ccc;")
        orig_vbox.addWidget(self.lbl_orig)
        self.group_orig.setLayout(orig_vbox)

        # Right Side: Scroll Area for Multiple Results
        self.group_res = QGroupBox("识别详情")
        res_vbox = QVBoxLayout()
        
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: white; border: none;")
        
        # 滚动区域的内容容器
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # 从上往下排
        self.scroll_content.setLayout(self.scroll_layout)
        
        self.scroll_area.setWidget(self.scroll_content)
        res_vbox.addWidget(self.scroll_area)
        self.group_res.setLayout(res_vbox)

        img_layout.addWidget(self.group_orig, 5) # 原图占 5/8
        img_layout.addWidget(self.group_res, 3)  # 列表占 3/8
        layout.addLayout(img_layout, 1) 

        # === 3. 任务列表 ===
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["文件名", "状态", "识别结果 (按置信度)", "颜色", "耗时"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) 
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows) 
        self.table.cellClicked.connect(self.on_table_click) 
        layout.addWidget(self.table, 1) 

        self.status = self.statusBar()

    # ================= 逻辑 =================

    def reset_ui(self):
        self.table.setRowCount(0)
        self.table_data = {}
        self.lbl_orig.setText("等待执行...")
        self.lbl_orig.setPixmap(QPixmap())
        # 清空滚动区
        self.clear_scroll_area()
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(False)

    def clear_scroll_area(self):
        """清空右侧详情列表"""
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def load_single_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.jpeg)")
        if not path: return
        self.reset_ui()
        self.add_task_to_table(path)
        self.btn_run.setEnabled(True)

    def load_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder: return
        self.reset_ui()
        folder_path = Path(folder)
        try:
            for p in sorted(folder_path.glob("*")):
                if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                    self.add_task_to_table(str(p))
        except: pass
        if self.table.rowCount() > 0: self.btn_run.setEnabled(True)

    def add_task_to_table(self, img_path):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(Path(img_path).name))
        self.table.setItem(row, 1, QTableWidgetItem("等待中"))
        self.table.setItem(row, 2, QTableWidgetItem("-"))
        self.table.setItem(row, 3, QTableWidgetItem("-"))
        self.table.setItem(row, 4, QTableWidgetItem("-"))
        self.table_data[row] = {'path': img_path, 'res_dir': None, 'results': None}

    def start_processing(self):
        if not os.path.exists(self.det_model):
            QMessageBox.critical(self, "错误", "模型文件缺失")
            return
        
        self.btn_run.setEnabled(False)
        self.progress_bar.setRange(0, self.table.rowCount())
        self.progress_bar.setVisible(True)

        tasks = []
        for row in range(self.table.rowCount()):
            tasks.append((row, self.table_data[row]['path']))
            self.table.setItem(row, 1, QTableWidgetItem("Running..."))

        self.worker = BatchWorker(tasks, self.det_model, self.rec_model, self.output_dir)
        self.worker.item_finished.connect(self.on_item_finished)
        self.worker.progress_update.connect(lambda c, t: self.progress_bar.setValue(c))
        self.worker.all_finished.connect(lambda: [self.btn_run.setEnabled(True), QMessageBox.information(self, "完成", "处理结束")])
        self.worker.start()

    def on_item_finished(self, row, img_path, res_dir, results_list, t_cost):
        # 1. 状态判断
        if results_list is None: 
            # 异常错误
            status_item = QTableWidgetItem("Error")
            status_item.setForeground(QBrush(QColor("red")))
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem("Exception"))
            return
        
        if len(results_list) == 0:
            # 未检测到车牌
            status_item = QTableWidgetItem("未检出")
            status_item.setForeground(QBrush(QColor("#E6A23C")))
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem("未检测到车牌"))
        else:
            # 成功检测 (可能 1 个，也可能 N 个)
            status_item = QTableWidgetItem(f"成功({len(results_list)})")
            status_item.setForeground(QBrush(QColor("green")))
            self.table.setItem(row, 1, status_item)
            
            # 拼接所有结果，用 " | " 分隔
            # 例如: "京A88888 | 京B12345"
            all_texts = [r['text'] for r in results_list]
            all_colors = [r['color'] for r in results_list]
            
            self.table.setItem(row, 2, QTableWidgetItem(" | ".join(all_texts)))
            self.table.setItem(row, 3, QTableWidgetItem(" | ".join(all_colors)))

        self.table.setItem(row, 4, QTableWidgetItem(f"{t_cost:.2f}s"))
        
        # 存数据
        self.table_data[row]['res_dir'] = res_dir
        self.table_data[row]['results'] = results_list

        # 如果只有一行，自动显示
        if self.table.rowCount() == 1:
            self.table.selectRow(row)
            self.display_result(row)

    def on_table_click(self, row, col):
        self.display_result(row)

    def display_result(self, row):
        """核心：动态渲染右侧的滚动列表"""
        data = self.table_data.get(row)
        if not data: return
        
        img_path = data['path']
        res_dir = data['res_dir']
        results = data['results'] # 这是一个列表

        # 1. 显示左侧大图 (带框图)
        if res_dir:
            labeled = Path(res_dir) / "detection_result" / "labeled_original.jpg"
            if labeled.exists():
                self.show_image(self.lbl_orig, str(labeled))
            else:
                self.show_image(self.lbl_orig, img_path)
        else:
            self.show_image(self.lbl_orig, img_path)

        # 2. 渲染右侧详情列表
        self.clear_scroll_area() # 先清空旧的

        if not results:
            # Case 1: 未检测到
            lbl = QLabel("未检测到车牌\n\n请检查图片质量")
            lbl.setStyleSheet("color: #E6A23C; font-size: 14px; font-weight: bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll_layout.addWidget(lbl)
            return

        # Case 2: 循环渲染每一个车牌卡片
        for idx, item in enumerate(results):
            # 创建一个卡片容器 (Frame)
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    margin-bottom: 5px;
                }
            """)
            card_layout = QHBoxLayout()
            
            # 左：矫正后的小图
            lbl_crop = QLabel()
            lbl_crop.setFixedSize(120, 50)
            lbl_crop.setStyleSheet("border: 1px dashed #999; bg-color: #eee;")
            lbl_crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            crop_path = item.get('crop_path', '')
            if crop_path and os.path.exists(crop_path):
                pix = QPixmap(crop_path)
                lbl_crop.setPixmap(pix.scaled(lbl_crop.size(), Qt.AspectRatioMode.KeepAspectRatio))
            else:
                lbl_crop.setText("无图像")
            
            # 右：文字信息
            info_text = f"""
            <div style='line-height:1.4;'>
                <span style='font-size:14px; font-weight:bold; color:#0078D7;'>{item['text']}</span><br>
                <span style='color:#666;'>颜色: {item['color']} | ID: {idx+1}</span>
            </div>
            """
            lbl_info = QLabel(info_text)
            lbl_info.setStyleSheet("border: none;") # 去掉内部label边框
            
            card_layout.addWidget(lbl_crop)
            card_layout.addWidget(lbl_info)
            card.setLayout(card_layout)
            
            # 添加到滚动列表
            self.scroll_layout.addWidget(card)

    def show_image(self, label, path):
        pix = QPixmap(path)
        if not pix.isNull():
            scaled = pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
        else:
            label.setText("无法加载")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())