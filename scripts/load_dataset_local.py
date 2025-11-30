# scripts/load_dataset_local.py
import os
from zipfile import ZipFile

# Path to your cached IndicAccentDB.zip
zip_path = r"C:\Users\ashwi\.cache\huggingface\hub\datasets--DarshanaS--IndicAccentDb\snapshots\d9e9a08f352843aa4bee542e408e4067893ead85\IndicAccentDB.zip"

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"❌ Dataset ZIP not found at:\n{zip_path}\n\n➡ Please check your Hugging Face cache folder.")

print("✅ Found dataset ZIP file!")
print(f"📦 Path: {zip_path}")

# Check contents
with ZipFile(zip_path, 'r') as zf:
    files = zf.namelist()
    print(f"\n✅ Total files: {len(files)}")
    print("📂 Top-level folders:")
    top_folders = sorted(set(f.split("/")[0] for f in files if "/" in f))
    for folder in top_folders:
        print(" -", folder)

    print("\n🎧 Example audio files:")
    for f in files[:10]:
        if f.endswith(".wav"):
            print(" ", f)
