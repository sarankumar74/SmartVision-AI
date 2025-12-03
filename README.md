# 🤖 SmartVision AI – Intelligent Multi-Class Object Recognition System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch)
![CNN](https://img.shields.io/badge/Architecture-CNN-yellow)
![YOLO](https://img.shields.io/badge/Object%20Detection-YOLOv8-green)
![TransferLearning](https://img.shields.io/badge/Method-Transfer%20Learning-purple)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/Deployment-HuggingFace-blue?logo=huggingface)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20AI-brightgreen)

---

## 📘 Overview
**SmartVision AI** is a next-generation **computer vision platform** that performs both **image classification** and **multi-object detection** across **25 diverse classes** derived from the **COCO dataset**.

The system integrates:
- **CNN-based Transfer Learning** (VGG16, ResNet50, MobileNet, EfficientNet)
- **YOLO-based Object Detection**
- **Model comparison dashboards**
- **Streamlit multipage Web Application**
- **Deployment on Hugging Face / Cloud**

This project demonstrates a **full deep learning lifecycle** — from **dataset preprocessing** to **real-time inference deployment**.

---

## 🎯 Problem Statement
Organizations across industries require reliable AI systems that can:
- ✔️ Detect **multiple objects in a single image**
- ✔️ Classify **objects across multiple categories**
- ✔️ Perform **real-time inference**
- ✔️ Maintain high accuracy in **different environments, lighting, and angles**
- ✔️ Scale for **cloud-based deployment**

To address this need, **SmartVision AI** combines image classification + object detection to build an intelligent, scalable, multi-domain solution.

---

## 💼 Business Use Cases

### 1️⃣ Smart Cities & Traffic Management
- Vehicle detection and counting  
- Pedestrian safety monitoring  
- Parking & lane rule violation alerts  

### 2️⃣ Retail & E-Commerce
- Automated product recognition  
- Scan-free checkout  
- Inventory tracking and planogram compliance  

### 3️⃣ Security & Surveillance
- Intrusion alerts  
- Suspicious object monitoring  
- Crowd density analytics  

### 4️⃣ Wildlife Conservation
- Automatic species recognition from camera traps  
- Poaching detection  
- Habitat monitoring  

### 5️⃣ Healthcare
- PPE compliance monitoring  
- Medical equipment recognition  
- Fall detection in hospitals  

### 6️⃣ Smart Home & IoT
- Home automation using object triggers  
- Pet monitoring and alert systems  

### 7️⃣ Agriculture
- Livestock counting  
- Harvest readiness identification  
- Pest/object detection in farmland  

### 8️⃣ Logistics & Warehousing
- Automated parcel sorting  
- Real-time inventory tracking  
- Damage detection on packages  

---

## 🧠 Skills Takeaway
- **Python for Deep Learning & Computer Vision**
- **TensorFlow / PyTorch for CNN model training**
- **Transfer Learning — VGG16, ResNet50, MobileNet, EfficientNet**
- **YOLO for Object Detection**
- **OpenCV for image preprocessing**
- **Model evaluation & confusion matrix analysis**
- **Streamlit Web App Development**
- **Hugging Face Cloud Deployment**

---

## ⚙️ Approach Summary

### 🔹 Step 1 — Dataset Preparation
- Used **curated subset of 25 classes from the COCO dataset**
- Normalized and resized images
- Applied augmentation: rotation, flip, brightness, zoom, blur

### 🔹 Step 2 — Image Classification (Transfer Learning)
Four CNN models were trained:
| Model | Type | Strength |
|--------|------|----------|
| VGG16 | Transfer Learning | Baseline benchmark |
| ResNet50 | Transfer Learning | Deep residual learning |
| MobileNet | Transfer Learning | Lightweight + fast |
| EfficientNet | Transfer Learning | High efficiency + accuracy |

Outputs:
- Top-1 & Top-5 class predictions
- Side-by-side model performance comparison

### 🔹 Step 3 — Object Detection (YOLO)
- YOLO model trained for **bounding box prediction + label + confidence**
- Optimized for **real-time inference & low latency**

### 🔹 Step 4 — Model Evaluation & Validation
- Accuracy / Precision / Recall / F1-Score
- Confusion Matrix
- FPS / inference time evaluation
- Class-wise performance breakdown

### 🔹 Step 5 — Streamlit Multi-Page Application
Includes:
1️⃣ **Home Page** – Overview & demo images  
2️⃣ **Image Classification** – Upload → get predictions from all 4 CNN models  
3️⃣ **Object Detection** – YOLO detection with bounding boxes  
4️⃣ **Model Performance Dashboard**  
5️⃣ **Live Webcam Detection (optional)**  
6️⃣ **About / Documentation Page**

### 🔹 Step 6 — Cloud Deployment
- Deployed on **Hugging Face Spaces / Streamlit Cloud**
- CI/CD enabled through GitHub

---

<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/d4ed0614-4b9e-4d31-9c60-6c94550c7c99)


#### Detection Results Page 1
![Result Page](https://github.com/user-attachments/assets/8e8884b4-db95-4fde-a077-7c14f82cd9f1)


####  Detection Results Page 2
![Dashboard](https://github.com/user-attachments/assets/c4acb123-7f31-48a7-8f2e-ea620dcce65b)


####  Detection Results Page 3
![Dashboard](https://github.com/user-attachments/assets/d7269bff-963b-4c16-9617-0ace8d8534a6)


## 🧩 Project Structure

```bash
SmartVision_AI/
│
├── datasets codes/
│   │                                      
│   └── Smart_Vision_Data_Code.ipynb /                    
│
├── Traninig Codes/
│   │
│   │
│   └── SmartVision_Train_Code.ipynb
│
├── classification/
│   ├── test/
│   │
│   └── train/
│
├── smart vision detection/
│   ├── train                
│   ├── valid                         
│   └── data.yaml                   
│
├── SmartVision_Train.ipynb/
│
│ 
├── app.py
│                      
└── requirements.txt
