# scripts/test_dataset.py
from datasets import load_dataset

print("🔄 Loading IndicAccentDb from Hugging Face...")
ds = load_dataset("DarshanaS/IndicAccentDb")

print(ds)
print("\n✅ Dataset loaded successfully!")
