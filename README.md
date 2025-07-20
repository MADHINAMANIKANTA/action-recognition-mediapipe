# action-recognition-mediapipe
# Real-Time Action Recognition in Multi-Person Pose Estimation Using MediaPipe

This project focuses on recognizing human actions in real time using multi-person pose estimation. We use **MediaPipe** to extract 2D skeletal landmarks, **YOLOv8** for multiple person detection, and train a **Random Forest** model to classify actions.

---

## 🎯 Recognized Actions
- 👊 Punching  
- 🙇 Sitting  
- 🙆 Standing  
- 🚶 Walking  
- 👋 Waving  

---

## 🧠 Model and Dataset

- **Model Used**: Random Forest Classifier  
- **Pose Estimation**: MediaPipe (BlazePose)  
- **Person Detection**: YOLOv8  
- **Dataset**: [N-UCLA RGB Dataset (Kaggle)](https://www.kaggle.com/datasets/akshayjain22/n-ucla-rgb)  
- **Accuracy Achieved**: ⭐ 86% on selected actions  

---

## 📁 Folder Structure
RealTime-Action-Recognition/
│
├── action_model.py # Main real-time recognition script
├── trained_model.pkl # Trained Random Forest model (if small enough)
├── sample_videos/ # Few test videos (optional)
├── requirements.txt # Python libraries used
└── README.md # This file

## ▶️ How to Run

```bash
pip install -r requirements.txt
python action_model.py
Make sure:

A webcam is connected.

The trained model is in the expected path.

YOLOv8 weights are available or downloaded via Ultralytics.
👨‍🏫 Guided by: Dr. G.R.S. Murthy

👨‍💻 Team Members

M. Manikanta
G. Mamatha Siri
P. Vasu
P. Varsha

📦 Requirements (partial)
opencv-python
mediapipe
ultralytics
scikit-learn
joblib
numpy
⚠️ Notes
Due to size, the full N-UCLA dataset and trained model are not included. Please refer to:

Dataset on Kaggle





