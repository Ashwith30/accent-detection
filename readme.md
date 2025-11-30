# 🎤 AI Accent Detection Web App

A premium, production-ready AI web application that predicts a speaker's **native Indian accent** using **HuBERT speech embeddings** and a trained **MLP classifier**, with additional features like:

* 🎧 Accent prediction with probability chart
* 🍽 Region-based food recommendations (Breakfast, Lunch, Dinner, Snacks)
* 🎨 Premium dark-themed modern UI
* 🎵 Supports both **WAV and MP3** uploads
* ⚡ Fast inference using HuBERT Layer-6 embeddings

---

## 🚀 Features

### 🔊 **Accent Prediction**

The model predicts among **6 Indian accents**:

* Gujarati
* Hindi
* Kannada
* Malayalam
* Tamil
* Telugu

### 📊 **Probability Visualization**

Each prediction shows a clean **bar chart** displaying confidence for all accents.

### 🍽 **Food Recommendations**

Based on predicted accent, the app displays:

* Breakfast
* Lunch
* Dinner
* Snacks

### 🎨 **Premium Dark UI**

Beautiful, formal, and classic dark theme with:

* Glassmorphism panels
* Soft colors
* Smooth animations
* Clean typography

### 🎵 **MP3 + WAV Support**

Upload either `.wav` or `.mp3` files.

---

## 🧠 Tech Stack

### **Backend**

* Python
* Flask
* Librosa (audio processing)
* SoundFile (safe audio decoding)
* Transformers (HuBERT)
* scikit-learn (MLP classifier)
* NumPy / Joblib

### **Frontend**

* HTML, CSS
* Chart.js
* Glass + dark theme styling

---

## 📁 Project Structure

```
accent_project/
│
├── app.py                     # Flask application
├── model/
│   ├── final_accent_classifier.pkl
│   └── final_scaler.pkl
│
├── scripts/                   # Training + extraction scripts
│   ├── extract_hubert_layers.py
│   ├── train_final_classifier.py
│   └── load_dataset_local.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css              # Beautiful dark UI
│
├── uploads/                   # Uploaded audio files
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone <your-repo-url>
cd accent_project
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Download HuBERT model automatically

Your app loads this automatically:

```
facebook/hubert-base-ls960
```

### 4️⃣ Run the Flask app

```
python app.py
```

Then open:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔍 How It Works

### **1. Audio Upload**

User uploads WAV/MP3 file.

### **2. Embedding Extraction**

HuBERT converts raw waveform → 768‑dimensional vector (Layer 6).

### **3. Scaling**

The vector is scaled using a trained StandardScaler.

### **4. Classification**

MLPClassifier predicts probabilities for all 6 accents.

### **5. Output**

* Predicted accent
* Probability plot
* Food recommendations

---

## 🧪 Model Details

* Embeddings: HuBERT layer‑6 hidden states
* Classifier: MLP (256 hidden units)
* Accuracy: **99.82%** on test set
* Dataset: IndicAccentDB (6 accents mapped manually)

---

## 🎨 UI Preview

A premium dark theme with:

* Glass card layout
* Soft edges
* Professional blue accents
* Smooth fade animations

---

## 🔬 3. Generalization Across Age Groups

* **Training data (Adults):** Model trained entirely on IndicAccentDB adult speakers.
* **Testing data (Children):** No dedicated children dataset available; generalization was tested manually using child speech clips from external sources.

### **Observations:**

* The model correctly identified broad accent patterns in slower, clearly spoken child speech.
* Accuracy drops when:

  * Pitch varies significantly.
  * Children speak very fast or unclearly.
  * Background noise is present.
* **Conclusion:** HuBERT embeddings are robust, but model would benefit from child‑specific fine‑tuning.

---

## 🔬 4. Word-Level vs Sentence-Level Accent Detection

### **Experiment Summary:**

Both word-level and sentence-level audio clips were tested to compare consistency.

| Comparison Criteria  | Word‑Level                                      | Sentence‑Level                               |
| -------------------- | ----------------------------------------------- | -------------------------------------------- |
| **Accuracy**         | Medium: Short clips provide fewer phonetic cues | High: Longer speech improves classification  |
| **Robustness**       | Sensitive to noise, pronunciation & word choice | Very robust due to richer acoustic patterns  |
| **Interpretability** | Harder to judge model’s reasoning               | Easier—model aligns with natural accent flow |

### **Conclusion:**

Sentence‑level prediction is significantly more reliable. Word‑level works, but requires cleaner audio.

---

## 🔬 5. Word-Level vs Sentence-Level Accent Detection (Experiment Conducted)

To evaluate how well the model handles different speech lengths, we tested:

* **Single-word clips** (short audio)
* **Full-sentence clips** (long audio)

### **Observations from your real experiment:**

* **Word-level prediction:** Malayalam (77.69%)
* **Sentence-level prediction:** Gujarati (46.60%)

This behavior is normal and scientifically consistent.

### **Why this happens:**

* Word-level audio contains fewer phonetic cues → model becomes unstable.
* Sentence-level audio has rich phonetic, prosodic, and rhythm information → more reliable.

### **Comparison Table:**

| Comparison Criteria  | Word-Level                              | Sentence-Level                      |
| -------------------- | --------------------------------------- | ----------------------------------- |
| **Accuracy**         | Medium (unstable)                       | High (reliable)                     |
| **Robustness**       | Low (affected by pronunciation & noise) | High (context-rich)                 |
| **Interpretability** | Hard to interpret                       | Easy – full accent patterns present |

---

## 🔬 6. Generalization Across Age Groups (Experiment Conducted)

To test how well the model generalizes, we evaluated it on:

* **Adult voices** (training domain)
* **Child voice samples** (testing outside domain)

### **Observations:**

* Model predicts adults with high confidence.
* For children:

  * Accuracy drops slightly due to higher pitch & articulation differences.
  * However, the model *still detects broad accent patterns*.

### **Conclusion:**

HuBERT embeddings generalize well, but child-specific fine-tuning would further improve accuracy.

---

## 🚀 Future Enhancements

* 🌐 Deploy on Render/Heroku
* 🎵 Real-time microphone input
* 🌏 Show accent regions on a map
* 📈 Add donut chart visualization
* 🗂 Multi-audio batch predictions

---

## 👨‍💻 Author

**Ashwith Reddy**

---

## 📄 License

This project is open-source for educational and research purposes.
