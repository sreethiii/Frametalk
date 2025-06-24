# 🧠 FrameTalk

**FrameTalk** is a web-based object detection app built with Flask that uses YOLO models (v3, v5, v8) to detect objects in images and announce the results using text-to-speech. It's designed to be accessible, fast, and flexible.

---

## 🚀 Features

- 🖼 Upload images and detect objects in real-time
- 🧠 Supports multiple YOLO models: `YOLOv3`, `YOLOv5`, `YOLOv8`
- 🔊 Text-to-speech (TTS) for detected objects
- 📁 Organized file structure for maintainability
- 🌐 Simple web interface using Flask & HTML templates

---

## 🗂 Folder Structure
Frametalk/
│
├── Frametalk.py # Main Flask application
├── requirements.txt # List of dependencies
│
├── static/ # CSS, JS, and static assets
├── templates/ # HTML templates
├── uploads/ # Temporary uploaded images
├── yolo_model/ # YOLO configuration and helper files
│
├── yolov3.weights # YOLOv3 model weights (not pushed to GitHub)
├── yolo5n.pt # YOLOv5n model weights (not pushed to GitHub)
├── yolo5s.pt
├── yolo5su.pt
├── yolov8m.pt / yolov8m.onnx

# 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sreethiii/Frametalk.git
   cd Frametalk
   ```

2. **Install dependencies:**
🧩 Dependencies
Flask

OpenCV (cv2)

PyTorch

Ultralytics

pyttsx3 (for TTS)

pandas, requests, seaborn, numpy


```bash
pip install -r requirements.txt
```

3. **Download YOLO model weights:**


4. **Run the app:**
   ```bash
   python Frametalk.py
   ```

## 🔊 How It Works
User uploads an image

App runs object detection using the selected YOLO model

Detected objects are shown with bounding boxes

Each object is also read aloud using pyttsx3
