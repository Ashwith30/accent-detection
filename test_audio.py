import numpy as np
import joblib
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# -------------------------------
# LOAD MODEL + SCALER + HUBERT
# -------------------------------
classifier = joblib.load("model/final_accent_classifier.pkl")
scaler = joblib.load("model/final_scaler.pkl")

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

ACCENTS = ["gujarati", "hindi", "kannada", "malayalam", "tamil", "telugu"]

# -------------------------------
# EXTRACT HUBERT EMBEDDING
# -------------------------------
def extract_embedding(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    emb = outputs.hidden_states[6].squeeze().mean(dim=0).numpy()
    return emb.reshape(1, -1)

# -------------------------------
# MAIN TEST FUNCTION
# -------------------------------
def test_audio(path):
    print(f"\n🔍 Testing audio file: {path}")

    emb = extract_embedding(path)
    emb_scaled = scaler.transform(emb)

    probs = classifier.predict_proba(emb_scaled)[0]
    idx = int(np.argmax(probs))
    accent = ACCENTS[idx]

    print("\n🎯 Predicted Accent:", accent)
    print("\n📊 Probabilities:")
    for name, p in zip(ACCENTS, probs):
        print(f"{name:10s} → {p*100:.2f}%")

# -------------------------------
# RUN TEST
# -------------------------------
if __name__ == "__main__":
    path = input("Enter path to audio file: ")
    test_audio(path)
