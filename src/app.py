# this file not being used anymore due to using different UI package and elements


# import sys
# import cv2
#
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel,
#     QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
#     QGroupBox
# )
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtCore import Qt
#
# # === IMPORT YOUR PIPELINE ===
# from plate_detection import crop_plate
# from ocr_reader import read_plate_text
# from state_classifier import classify_state
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.image_path = None
#
#         self.setWindowTitle("IPPR – License Plate Recognition")
#         self.setGeometry(100, 100, 1200, 700)
#
#         self.init_ui()
#
#     def init_ui(self):
#         main_widget = QWidget()
#         main_layout = QHBoxLayout()
#
#         # ========== SIDEBAR ==========
#         sidebar = QVBoxLayout()
#
#         btn_load = QPushButton("📂 Load Image")
#         btn_run = QPushButton("▶ Run Detection")
#
#         sidebar.addWidget(btn_load)
#         sidebar.addWidget(btn_run)
#         sidebar.addStretch()
#
#         sidebar_widget = QWidget()
#         sidebar_widget.setLayout(sidebar)
#         sidebar_widget.setFixedWidth(180)
#
#         # ========== CONTENT ==========
#         content = QVBoxLayout()
#
#         title = QLabel("Malaysian License Plate Recognition System")
#         title.setAlignment(Qt.AlignCenter)
#         title.setStyleSheet("font-size:18px; font-weight:bold;")
#
#         content.addWidget(title)
#
#         grid = QHBoxLayout()
#
#         self.input_image = QLabel("Input Image")
#         self.input_image.setAlignment(Qt.AlignCenter)
#         self.input_image.setStyleSheet("border:1px solid gray;")
#
#         self.plate_image = QLabel("Detected Plate")
#         self.plate_image.setAlignment(Qt.AlignCenter)
#         self.plate_image.setStyleSheet("border:1px solid gray;")
#
#         grid.addWidget(self.wrap_box("Input Image", self.input_image))
#         grid.addWidget(self.wrap_box("Plate Region", self.plate_image))
#
#         content.addLayout(grid)
#
#         self.result_label = QLabel("Plate: —\nState: —")
#         self.result_label.setAlignment(Qt.AlignCenter)
#         self.result_label.setStyleSheet("font-size:16px;")
#
#         content.addWidget(self.wrap_box("OCR Result", self.result_label))
#
#         # Combine layouts
#         main_layout.addWidget(sidebar_widget)
#         main_layout.addLayout(content)
#
#         main_widget.setLayout(main_layout)
#         self.setCentralWidget(main_widget)
#
#         # Connect buttons
#         btn_load.clicked.connect(self.load_image)
#         btn_run.clicked.connect(self.run_detection)
#
#     def wrap_box(self, title, widget):
#         box = QGroupBox(title)
#         layout = QVBoxLayout()
#         layout.addWidget(widget)
#         box.setLayout(layout)
#         return box
#
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
#         )
#
#         if not path:
#             return
#
#         self.image_path = path
#
#         pixmap = QPixmap(path).scaled(
#             400, 250, Qt.KeepAspectRatio
#         )
#         self.input_image.setPixmap(pixmap)
#
#         self.plate_image.clear()
#         self.result_label.setText("Image loaded.\nClick 'Run Detection'.")
#
#     def run_detection(self):
#         if not self.image_path:
#             self.result_label.setText("Please load an image first.")
#             return
#
#         plate_img = crop_plate(self.image_path)
#
#         if plate_img is None:
#             self.result_label.setText("Plate not detected.")
#             return
#
#         # Show detected plate
#         plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = plate_rgb.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(
#             plate_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
#         )
#
#         self.plate_image.setPixmap(
#             QPixmap.fromImage(qt_image).scaled(
#                 400, 200, Qt.KeepAspectRatio
#             )
#         )
#
#         # OCR + State
#         plate_text = read_plate_text(plate_img)
#         state = classify_state(plate_text)
#
#         self.result_label.setText(
#             f"Plate: {plate_text if plate_text else '—'}\nState: {state}"
#         )
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())
