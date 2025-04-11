# 🌿 Smart Plant Disease Detection and Reporting System

This project is a **final year engineering project** focused on developing a smart system that detects plant diseases using machine learning and reports them via a Telegram bot. It combines **computer vision**, **ESP32-CAM**, **robotic mobility**, and **IoT** to provide farmers with real-time insights on plant health.

---

## 🔍 Project Overview

The system captures images of plant leaves using an **ESP32-CAM module** mounted on a mobile robot. These images are then sent to a **Flask server**, where a **convolutional neural network (CNN)** model trained on the PlantVillage dataset analyzes them to detect whether the plant is healthy or diseased. If a disease is detected, a notification is sent instantly via **Telegram**.

---

## 📦 Dataset

We used the **PlantVillage Dataset**, which contains thousands of labeled images of healthy and diseased plant leaves.

📥 Download the dataset here:  
👉 [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage/code)

---

## 🖼️ Image Previews

### 🔧 Hardware Prototype

This is what the hardware robot with the ESP32-CAM looks like:

![Hardware device](Sample.jpeg)

### 📲 Telegram Notification

Below is a preview of the Telegram notification received when a plant disease is detected:

![Telegram notification preview](Sample2.jpeg)

---

## 🚀 Features

- ESP32-CAM captures live plant images
- Sends images to Flask server via HTTP
- CNN model classifies the plant health status
- Telegram bot sends instant disease alerts
- Ultrasonic sensors allow autonomous mobility

---

Feel free to fork, contribute, and improve this project!
