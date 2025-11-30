# scripts/test_setup.py
import torch, librosa, transformers, sklearn, flask, joblib

print("✅ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
