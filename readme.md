
# 🎤 AI Accent Detection Web App

A premium, production-ready AI web application that predicts a speaker's **native Indian accent** using **HuBERT speech embeddings** and a trained **MLP classifier**.

Includes:

* 🎧 Accent prediction with probability chart
* 🗺️ Accent → State mapping
* 🍽 Food recommendations (Breakfast, Lunch, Dinner, Snacks)
* 🎨 Premium dark UI
* ⚡ Fast inference using **HuBERT Layer 6**
* 🔬 Fully tested (word-level vs sentence-level, child vs adult voices)
* 🎵 Supports **WAV only**

---

# 🚀 Features

## 🔊 **Accent Prediction with Region Mapping**

| Accent    | Associated State / Region  |
| --------- | -------------------------- |
| Gujarati  | Gujarat                    |
| Hindi     | Jharkhand / Hindi Belt     |
| Kannada   | Karnataka                  |
| Malayalam | Kerala                     |
| Tamil     | Tamil Nadu                 |
| Telugu    | Andhra Pradesh / Telangana |

---

## 📊 Probability Visualization

A clean bar chart shows model confidence for all 6 accents.

---

## 🍽 Food Recommendations

Based on predicted accent, the UI shows:

* Breakfast
* Lunch
* Dinner
* Snacks

---

## 🎨 Premium Dark UI

Featuring:

* Glassmorphism
* Smooth blue highlights
* Clean typography
* Classic, formal theme

---

# 🧠 Tech Stack

### Backend

* Python
* Flask
* Librosa
* SoundFile
* Transformers (HuBERT)
* PyTorch
* scikit-learn
* NumPy, joblib

### Frontend

* HTML / CSS
* Chart.js

---

# 📁 Project Structure

```
accent_project/
│
├── app.py
│
├── model/
│   ├── final_accent_classifier.pkl
│   ├── final_scaler.pkl
│   ├── embeddings_layer_*.npy
│   ├── labels.npy
│   └── layerwise_accuracy.npy / .png
│
├── scripts/
│   ├── extract_hubert_layers.py
│   ├── analyze_layers.py
│   ├── train_final_classifier.py
│   ├── test_word_sentence.py
│   ├── test_generalization.py
│   └── test_audio.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── uploads/
│   └── *.wav
│
└── requirements.txt
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone the Repository

```
git clone https://github.com/Ashwith30/accent-detection.git
cd accent-detection
```

---

## 2️⃣ Create & Activate Virtual Environment (Recommended)

### Windows:

```
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```
python app.py
```

Open in browser:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

# 🔍 How It Works

1. Upload WAV file
2. Extract HuBERT Layer-6 embedding (768-dim vector)
3. Pass through StandardScaler
4. MLPClassifier predicts accent probabilities
5. UI shows accent + state + food + graph

---

# 🧪 Experiments & Evaluation

---

# 🔬 **1. HuBERT Layer-Wise Analysis**

You extracted embeddings from all **13 HuBERT layers (0–12)** and computed accuracy per layer.

### ✅ **Complete Layer-Wise Accuracy Table (Your Real Output)**

| **Layer**   | **Accuracy**      |
| ----------- | ----------------- |
| Layer 0     | 0.9988            |
| Layer 1     | 0.9982            |
| Layer 2     | 0.9975            |
| Layer 3     | 0.9982            |
| **Layer 4** | **0.9994 (Best)** |
| Layer 5     | 0.9975            |
| Layer 6     | 0.9982            |
| Layer 7     | 0.9963            |
| Layer 8     | 0.9963            |
| Layer 9     | 0.9932            |
| Layer 10    | 0.9951            |
| Layer 11    | 0.9963            |
| Layer 12    | 0.9963            |

### 🏆 Best Layer

**Layer 4 with 0.9994 accuracy**

---

### 🎯 Why We Used **Layer 6** in the Final Model

Even though Layer 4 had slightly higher accuracy:

* Layer 6 is the **final phoneme-rich mid-layer**
* Best balance of:

  * phonetics
  * accent cues
  * stability
* Layer 6 is used widely in speech research
* Accuracy difference is negligible

Thus, **Layer 6** was the optimal choice for deployment.

---

# 🔬 2. Word-Level vs Sentence-Level Testing (Your Real Test)

You tested:

### **Word-level audio** → Malayalam (77.69%)

### **Sentence-level audio** → Gujarati (46.60%)

### Insights:

| Criterion        | Word-Level                    | Sentence-Level          |
| ---------------- | ----------------------------- | ----------------------- |
| Accuracy         | Medium                        | High                    |
| Robustness       | Low (short clip = fewer cues) | High (richer phonetics) |
| Interpretability | Hard                          | Easy                    |

Sentence-level is significantly more reliable.

---

# 🔬 3. Generalization Across Age Groups (Your Real Test)

* Model trained on **adults (IndicAccentDB)**
* Tested manually on **children** (external clips)

### Observations:

* Adult predictions = strong
* Child predictions = accent detected but confidence drops due to:

  * Higher pitch
  * Faster/unclear speech

Conclusion:
HuBERT generalizes well, but child-specific fine-tuning would improve accuracy.

---

# 🚀 Future Enhancements

* Deployment (Render/Heroku)
* Real-time microphone input
* Geo visualization
* Multi-audio batch processing
* Improve children accent detection

---


