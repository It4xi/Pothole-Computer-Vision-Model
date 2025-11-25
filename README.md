# 🚗💡 Pothole-Computer-Vision-Model  
### Real-Time Object Detection for Safer Roads

This project uses a **YOLOv8 model converted to TensorFlow Lite** for detecting potholes in **real-time** from a camera feed.  
Designed for **lightweight deployment** and optimized to run at **~20 FPS** on standard laptops.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🧠 AI Detection | YOLOv8 model fine-tuned for accurate pothole detection |
| ⚡ Fast Inference | TensorFlow Lite for optimized model execution |
| 🎥 Real-Time Feed | Works with webcam or external video files |
| 📦 Efficient | NMS applied to remove overlapping boxes |
| 📊 Performance Stats | FPS displayed live on-screen |

---

## 🧰 Tech Stack

- Python 3.x  
- OpenCV  
- NumPy  
- TensorFlow Lite  
- YOLOv8 (exported to `.tflite`)

---

## 📁 Project Structure

📦 Pothole-Computer-Vision-Model
┣ 📂 notebook/
┣ 📜 main.py
┣ 📜 README.md
┣ 📜 LICENSE
┗ 📦 Yolov8-fintuned-on-potholes_int8.tflite


---

## 🛠 Installation

### 1️⃣ Clone the Repository
git clone https://github.com/lt4xi/Pothole-Computer-Vision-Model.git
cd Pothole-Computer-Vision-Model

2️⃣ Install Dependencies
pip install opencv-python numpy tensorflow


If using GPU-enabled TensorFlow, install the version compatible with your CUDA.

3️⃣ Add the Model

Place your .tflite model file inside the project folder:

Yolov8-fintuned-on-potholes_int8.tflite

▶️ Running the Application
Live webcam detection
python main.py


Press q to exit the live window.

🎥 Run on a Video File (optional)

In main.py change:

if __name__ == "__main__":
    run_detection(video_source="test_video.mp4")


Then execute again:

python main.py

🔧 Model Settings (Editable)

Inside main.py:

MODEL_PATH  = "Yolov8-fintuned-on-potholes_int8.tflite"
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
IMG_SIZE    = 320

⚙️ How It Works — Pipeline

📷 Webcam Frame
        ↓
🖼 Preprocessing (Resize, Normalize)
        ↓
🤖 YOLOv8 TFLite Inference
        ↓
📐 Bounding Box Extraction
        ↓
🚫 NMS Filtering
        ↓
🖊 Draw Detections + FPS
        ↓
🪟 Display Output Live


Developed with ❤️ by Kartheek (lt4xi)
B.Tech AI & ML — Computer Science & Engineering
