# 👗 FitBoard

FitBoard is a fashion mood board application that recommends visually similar outfits based on user-uploaded clothing items. It uses clustering (GMM) and feature similarity to generate outfit inspiration.

---

## 🚀 Setup Instructions

### 1. Clone the repository
git clone <your-repo-link>  
cd <your-repo-folder>

---

### 2. Install dependencies
pip install -r requirements.txt

---

### 3. Add HuggingFace Token

Create a `.env` file in the root directory and add:

HF_TOKEN=your_huggingface_token_here

You must have access to the dataset:  
https://huggingface.co/datasets/mvasil/polyvore-outfits

---

## ⚠️ IMPORTANT — Download Dataset First

Before running the app, you MUST download the outfit item images.

Run:
python download_images.py

This will:
- Download required images from HuggingFace  
- Save them in the `images/` folder  

⏱ This may take a few minutes (first time only)

---

## ▶️ Run the App
python -m streamlit run app.py

Then open in your browser:
http://localhost:8501

---

## ✨ Features

- Upload clothing items  
- Predict outfit style cluster  
- Generate similar outfit recommendations  
- Pinterest-style visual layout  
- Local image caching for fast performance  

---

## 🧠 Tech Stack

- Python  
- Streamlit  
- Scikit-learn (GMM, PCA)  
- OpenCV / Skimage (feature extraction)  
- HuggingFace Datasets  

---

## 📌 Notes

- The `images/` folder is not included in the repository  
- It must be generated using `download_images.py`  

---

## 👤 Author

FitBoard — Machine Learning Fashion Recommendation System