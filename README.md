# ConvNets-in-PyTorch
# 🚀 Convolutional Neural Networks (CNNs) for CIFAR-10 Classification using PyTorch

## 📌 Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset into 10 object categories.

The goal is to build a scalable, modular deep learning pipeline using modern computer vision best practices including convolutional layers, batch normalization, dropout, and GPU acceleration.

---

## 🎯 Objectives
- Implement CNN architecture in PyTorch
- Train and evaluate on CIFAR-10 dataset
- Apply data normalization and batching
- Optimize training using Adam/SGD
- Analyze model performance on multi-class classification

---

## 🧠 Model Architecture
- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Fully Connected Layers
- Softmax Output (10 classes)

---

## 📂 Dataset
CIFAR-10 Dataset:
- 60,000 32×32 color images
- 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

---

## ⚙️ Tech Stack
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib

---

## 🚀 How to Run

### 1️⃣ Install dependencies
pip install torch torchvision matplotlib


### 2️⃣ Run training script


python cifar10_cnn.py


---

## 📊 Results
- Achieved competitive accuracy on CIFAR-10 test dataset
- Demonstrated effective spatial feature extraction
- Showed improved performance over fully connected networks

---

## 📈 Key Learnings
- Convolutional feature extraction
- Multi-class classification with CrossEntropyLoss
- DataLoader pipelines
- GPU training workflow

---

## 👨‍💻 Author
Ritik Tomar  
