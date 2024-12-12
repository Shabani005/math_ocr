import json
import os
import sys
import subprocess
import torch
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QFrame, QPushButton
from PIL import Image
import time
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor

# Global variable to store the cached model
cached_model = None

folder_name = "images"

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# Utility to convert QImage to PIL Image
def QImageToPILImage(qimage):
    buffer = qimage.bits().asstring(qimage.byteCount())
    image = Image.frombytes("RGBA", (qimage.width(), qimage.height()), buffer)
    return image.convert("RGB")

class RegionSelector(QMainWindow):
    region_selected = pyqtSignal(QRect)

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Screen Region Selector")
        self.showFullScreen()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowOpacity(0.5)
        self.setStyleSheet("background-color: rgba(30, 30, 30, 0.9);")
        self.start_point = None
        self.end_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            selected_region = QRect(self.start_point, self.end_point).normalized()
            self.region_selected.emit(selected_region)
            self.close()

    def paintEvent(self, event):
        if self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setPen(Qt.red)
            painter.drawRect(QRect(self.start_point, self.end_point))

class ImageProcessor(QThread):
    finished = pyqtSignal(str)

    def __init__(self, image_path, device, model=None):
        super().__init__()
        self.image_path = image_path
        self.device = device
        self.model = model  # Reuse model if passed

    def run(self):
        global cached_model

        # Use pre-loaded model
        model = self.model if self.model else cached_model

        # Start the timer for inference
        start_time = time.time()

        # Load processor
        processor = load_processor()

        # Open the image
        img = Image.open(self.image_path).convert("RGB")

        # Perform inference
        results = batch_inference([img], model, processor)

        # End the timer
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        # Emit results
        self.finished.emit(str(results))

class ScreenCapturer(QMainWindow):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Screen Capturer")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Select a region to capture")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff;")
        self.capture_button = QPushButton("Capture Screen")
        self.capture_button.setStyleSheet("background-color: #444; color: #fff; border: none; padding: 10px; border-radius: 5px;")
        self.capture_button.clicked.connect(self.select_region)

        layout.addWidget(self.label)
        layout.addWidget(self.capture_button)

        self.setCentralWidget(frame)

        self.image_folder = "images"
        os.makedirs(self.image_folder, exist_ok=True)

        # Pre-load the model before user interaction
        self.model = self.load_model()

    def load_model(self):
        """Load the model once during initialization."""
        global cached_model
        if cached_model is None:
            print("Loading model...")
            cached_model = load_model()
            cached_model = cached_model.to("cuda" if torch.cuda.is_available() else "cpu")
            print("Model loaded and cached.")
        return cached_model

    def select_region(self):
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self.capture_region)
        self.selector.show()

    def capture_region(self, region):
        x, y, w, h = region.x(), region.y(), region.width(), region.height()
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0, x, y, w, h).toImage()
        captured_image = QImageToPILImage(screenshot)

        image_path = os.path.join(self.image_folder, "captured_image.png")
        captured_image.save(image_path)

        pixmap = QPixmap.fromImage(screenshot)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
        self.label.setText(f"Image saved to {image_path}")

        self.process_captured_image(image_path)

    def process_captured_image(self, image_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor_thread = ImageProcessor(image_path, device, model=self.model)
        self.processor_thread.finished.connect(self.display_inference_results)
        self.processor_thread.start()

    def display_inference_results(self, results):
        self.label.setText(f"Result copied to ClipBoard")
        clipboard = QApplication.clipboard()
        clipboard.setText(results)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    capturer = ScreenCapturer()
    capturer.show()
    sys.exit(app.exec_())
