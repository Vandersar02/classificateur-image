# Image Classifier with Google Image Search

This Python project allows users to load an image, classify it using a pre-trained MobileNetV2 model, and display a related image fetched from Google Image Search.

## Features

- Load an image from the file system
- Classify the loaded image using MobileNetV2
- Fetch and display a related image from Google Image Search
- Display classification results and images in a graphical user interface

## Requirements

- Python 3.6+
- TensorFlow
- NumPy
- Requests
- Pillow
- Google Images Search
- Tkinter (included with standard Python installation)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Vandersar02/image-classifier.git
   cd image-classifier
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```sh
   python main.py
   ```

2. Click the "..load image" button to select and classify an image from your file system.

3. The classification results will be displayed, and a related image will be fetched and displayed from Google Image Search.

## Configuration

To use Google Image Search, you need to provide your own API key and CX in the `load_image` function in line (93 - 94) of image-classifier.py:
```python
api_key = 'YOUR_GOOGLE_API_KEY'
cx = 'YOUR_CX'
```
