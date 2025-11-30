# scripts/extract_hubert_layers.py
# --------------------------------------
# HuBERT Layer-wise Embedding Extraction
# --------------------------------------
import os
import torch
import librosa
import numpy as np
from zipfile import ZipFile
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# -------------------------------------------------------
# 1️⃣ Path to dataset ZIP (update if needed)
# -------------------------------------------------------
zip_path = r"C:\Users\ashwi\.cache\huggingface\hub\datasets--DarshanaS--IndicAccentDb\snapshots\d9e9a08f352843aa4bee542e408e4067893ead85\IndicAccentDB.zip"

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"❌ Dataset ZIP not found:\n{zip_path}")

# -------------------------------------------------------
# 2️⃣ Load HuBERT model
# -------------------------------------------------------
print("🎯 Loading HuBERT model...")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model.eval()

# -------------------------------------------------------
# 3️⃣ Extract .wav files directly from ZIP
# -------------------------------------------------------
embeddings_by_layer = [[] for _ in range(13)]
labels = []

accent_map = {
    "andhra_pradesh": "telugu",
    "tamil": "tamil",
    "kerala": "malayalam",
    "karnataka": "kannada",
    "gujrat": "gujarati",
    "jharkhand": "hindi"
}

print("🎧 Extracting embeddings layer-wise...")
with ZipFile(zip_path, "r") as zf:
    all_files = [f for f in zf.namelist() if f.endswith(".wav")]
    print(f"✅ Total audio files: {len(all_files)}")

    # For testing, limit sample count (set to None for all)
    limit = None  # try 300 first if slow
    if limit:
        all_files = all_files[:limit]

    for file in tqdm(all_files):
        try:
            accent_folder = file.split("/")[0]
            accent = accent_map.get(accent_folder, None)
            if not accent:
                continue

            # Read WAV file bytes
            wav_bytes = zf.read(file)
            tmp_path = "temp_audio.wav"
            with open(tmp_path, "wb") as tmp_f:
                tmp_f.write(wav_bytes)

            # Load & resample
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            input_values = torch.tensor(y).unsqueeze(0)

            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.hidden_states  # tuple of 13 tensors

            # Average pooling for each layer
            for i in range(13):
                layer_emb = hidden_states[i].mean(dim=1).squeeze().numpy()
                embeddings_by_layer[i].append(layer_emb)

            labels.append(accent)
            os.remove(tmp_path)

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

# -------------------------------------------------------
# 4️⃣ Save embeddings
# -------------------------------------------------------
print("\n💾 Saving extracted embeddings...")

os.makedirs("model", exist_ok=True)
for i in range(13):
    np.save(f"model/embeddings_layer_{i}.npy", np.array(embeddings_by_layer[i], dtype=np.float32))

np.save("model/labels.npy", np.array(labels))
print("✅ All embeddings saved successfully!")
print("✅ Files stored in ./model/")
 
print("\n✅ Layers extracted: 0–12")
print("Now you can train & compare accuracy layer by layer.")
