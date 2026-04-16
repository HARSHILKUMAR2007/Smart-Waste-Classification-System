# ♻️ Smart Waste Classification System

An AI-powered web application that classifies waste images into 12 categories using a Convolutional Neural Network (CNN) based on **MobileNetV2** and provides eco-friendly disposal suggestions.

---

## 🚀 Project Overview

This project aims to promote **sustainability and proper waste management** by helping users identify waste types and dispose of them responsibly.

* 🔍 Upload an image of waste
* 🤖 AI model predicts the category
* 🌍 Get eco-friendly disposal tips

---

## 🧠 Model Details

* **Architecture:** MobileNetV2 (Transfer Learning)
* **Technique:** CNN + Fine-tuning
* **Input Size:** 224 × 224 pixels
* **Dataset:** Kaggle Garbage Classification Dataset
* **Classes:** 12 waste categories
* **Final Accuracy:** **92.11%**

---

## 📂 Dataset

Dataset used:
👉 https://www.kaggle.com/datasets/mostafaabla/garbage-classification

* ~15,000+ images
* Categories include:

  * Plastic
  * Paper
  * Metal
  * Glass (multiple types)
  * Clothes
  * Shoes
  * Battery
  * Biological waste
  * Trash

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** PIL, NumPy

---

## 📁 Project Structure

```
project/
│── app.py
│── model/
│   ├── model.h5
│   └── class_indices.json
│── train.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/waste-classification.git
cd waste-classification
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
streamlit run app.py
```

### 5️⃣ Open in browser

```
http://localhost:8501
```

---

## 📸 Usage

* Upload an image of a waste item
* The model predicts its category
* View confidence score and eco-friendly tips

---

## 📊 Model Performance

* **Test Accuracy:** 92.11%
* Strong performance on:

  * Clothes, Shoes, Battery
* Slight confusion in:

  * Plastic vs Metal
  * Glass categories

---

## 🌱 Sustainability Impact

* Encourages proper waste segregation
* Promotes recycling awareness
* Helps reduce landfill waste
* Supports eco-friendly practices

---

## ⚠️ Limitations

* Works best with **single-object images**
* Performance may drop for:

  * Blurry images
  * Mixed waste scenes
* Some classes visually overlap

---

## 🚀 Future Improvements

* Real-time camera integration
* Mobile app version
* Larger and more diverse dataset
* Improved classification for mixed waste
* Cloud deployment

---

## 👨‍💻 Author

**Harshil Patel**
B.Tech AI & ML Student

---

## ⭐ Acknowledgements

* Kaggle Dataset Contributors
* TensorFlow & Keras
* Streamlit

---

## 📜 License

This project is for educational purposes.
