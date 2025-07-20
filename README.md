# Real-Time Action Recognition in Multi-Person Pose Estimation Using MediaPipe
This project focuses on recognizing human actions in real time using multi-person pose estimation. We use **MediaPipe** to extract 2D skeletal landmarks, **YOLOv8** for multiple person detection, and train a **Random Forest** model to classify actions.
## Recognized Actions

- Punching  
- Sitting  
- Standing  
- Walking  
- Waving

## Dataset & Model

- **Dataset**: N-UCLA RGB Dataset ([Kaggle Link](https://www.kaggle.com/datasets/akshayjain22/n-ucla-rgb))
- **Model**: Random Forest Classifier
- **Pose Estimation**: MediaPipe
- **Detection**: YOLOv8
- **Accuracy**: 86%

## Folder Structure

RealTime-Action-Recognition/
├── action_model.py # Main Python script
├── trained_model.pkl # Trained model file (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

#Requirements:

Python 3.8+

A webcam connected

YOLOv8 weights downloaded

Trained Random Forest model in the correct path

#Guide: Dr. G.R.S. Murthy

#Team Members:
M. Manikanta

G. Mamatha Siri

P. Vasu

P. Varsha

#Dependencies:

opencv-python
mediapipe
ultralytics
scikit-learn
joblib
numpy

