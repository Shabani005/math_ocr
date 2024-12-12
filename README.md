# Statistics and Mathematics image to LaTeX OCR/Capturer

This project is a Python-based screen capturing and processing tool built with PyQt5 and Torch. The application allows users to select a region of the screen, capture the image, and process it using a pre-trained model. Then results are automatically copied to the clipboard.

## Features

- Capture a region of the screen using a graphical selection tool.
- Process the captured image using a pre-trained model.
- Use GPU acceleration with CUDA for faster processing (optional).

## Requirements

Before running the application, you need to install the required dependencies.

### Dependencies (required dependencies are automatically installed at runtime)

- **Python 3.7+**
- **PyQt5** for the graphical interface 
- **Torch** (PyTorch) for model inference 
- **texify** for model loading and processing
- **Pillow** for image handling
- **CUDA** for GPU acceleration (optional)

### Install Dependencies

1. Clone this repository:

   ```bash
   git clone https://github.com/Shabani005/math_ocr.git
   cd screen-capturer
