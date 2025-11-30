from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch
import os

app = Flask(__name__)

# -------------------------------
# Load model, scaler, processor
# -------------------------------
MODEL_PATH = "model/final_accent_classifier.pkl"
SCALER_PATH = "model/final_scaler.pkl"

classifier = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

ACCENTS = ["gujarati", "hindi", "kannada", "malayalam", "tamil", "telugu"]


FOOD_RECOMMENDATIONS = {
    "gujarati": {
        "breakfast": ["Thepla", "Khaman Dhokla", "Fafda Jalebi"],
        "lunch": ["Undhiyu", "Gujarati Kadhi", "Bajra Rotla"],
        "dinner": ["Dal Dhokli", "Sev Tamatar Sabzi", "Handvo"],
        "snacks": ["Ganthiya", "Khakhra", "Muthiya"]
    },
    "hindi": {
        "breakfast": ["Aloo Paratha", "Poha", "Paneer Sandwich"],
        "lunch": ["Rajma Chawal", "Dal Makhani", "Chole Bhature"],
        "dinner": ["Chapati & Sabzi", "Paneer Butter Masala", "Kadhi Chawal"],
        "snacks": ["Samosa", "Kachori", "Bhel Puri"]
    },
    "kannada": {
        "breakfast": ["Benne Dosa", "Idli Vada", "Ragi Mudde"],
        "lunch": ["Bisi Bele Bath", "Jowar Roti", "Kosambari"],
        "dinner": ["Akki Roti", "Neer Dosa", "Puliyogare"],
        "snacks": ["Mangalore Buns", "Kodubale", "Nippattu"]
    },
    "malayalam": {
        "breakfast": ["Appam", "Puttu & Kadala Curry", "Idiyappam"],
        "lunch": ["Kerala Sadhya", "Fish Curry Meal", "Avial"],
        "dinner": ["Malabar Parotta", "Vegetable Stew", "Chicken Roast"],
        "snacks": ["Banana Chips", "Unniyappam", "Pazham Pori"]
    },
    "tamil": {
        "breakfast": ["Idli Sambar", "Ven Pongal", "Kambu Dosai"],
        "lunch": ["Curd Rice", "Sambar Rice", "Rasam Rice"],
        "dinner": ["Kothu Parotta", "Onion Uttapam", "Millet Upma"],
        "snacks": ["Murukku", "Sundal", "Medhu Vada"]
    },
    "telugu": {
        "breakfast": ["Pesarattu", "Ghee Upma", "Idli & Karam Podi"],
        "lunch": ["Pulihora", "Gongura Mutton", "Perugu Annam"],
        "dinner": ["Ragi Sangati", "Kodi Kura", "Tomato Pappu"],
        "snacks": ["Punugulu", "Mirchi Bajji", "Garelu"]
    }
}



# -------------------------------
# Extract HuBERT Layer 6 Embedding
# -------------------------------
def extract_hubert_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)
    emb = outputs.hidden_states[6].squeeze().mean(dim=0).cpu().numpy()
    return emb.reshape(1, -1)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["audio"]
    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)

    emb = extract_hubert_embedding(path)
    emb_scaled = scaler.transform(emb)

    probs = classifier.predict_proba(emb_scaled)[0]
    accent_idx = int(np.argmax(probs))
    accent = ACCENTS[accent_idx]

# Convert to Python float for JSON compatibility
    results = [(ACCENTS[i], float(probs[i] * 100)) for i in range(len(ACCENTS))]

    foods = FOOD_RECOMMENDATIONS[accent]

    return render_template(
        "result.html",
        accent=accent.capitalize(),
        results=results,
        foods=foods,
        audio_file=file.filename
    )

if __name__ == "__main__":
    app.run(debug=False)

