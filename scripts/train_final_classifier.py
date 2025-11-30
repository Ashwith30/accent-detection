# scripts/train_final_classifier.py
# ------------------------------------------------------
# Train Final Accent Classifier (with probability output)
# ------------------------------------------------------
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 70)
print("🎯 STEP 4: Training Final Accent Classifier (HuBERT Layer 6)")
print("=" * 70)

# ------------------------------------------------------
# Load embeddings (Layer 6) and labels
# ------------------------------------------------------
X = np.load("model/embeddings_layer_6.npy", allow_pickle=True)
y = np.load("model/labels.npy", allow_pickle=True)
print(f"✅ Loaded embeddings: {X.shape}")
print(f"✅ Loaded labels: {len(y)} samples\n")

# ------------------------------------------------------
# Scale + Train/Test split
# ------------------------------------------------------
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------------------------------
# Evaluate
# ------------------------------------------------------
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

acc = accuracy_score(y_test, preds)
print(f"✅ Accuracy: {acc:.4f}\n")
print("📋 Classification Report:")
print(classification_report(y_test, preds))

# ------------------------------------------------------
# Save model + scaler
# ------------------------------------------------------
joblib.dump(clf, "model/final_accent_classifier.pkl")
joblib.dump(scaler, "model/final_scaler.pkl")

print("\n💾 Saved model/final_accent_classifier.pkl")
print("💾 Saved model/final_scaler.pkl")

# ------------------------------------------------------
# Test sample probability output + Debug checks
# ------------------------------------------------------
print("\n🎯 Sample probability prediction:")

accent_names = sorted(list(set(y)))  # must be 6 accents

sample_idx = 0
sample_probs = probs[sample_idx]

# IMPORTANT DEBUG CHECK
print("\n🔍 DEBUG CHECK:")
print("Classifier learned classes:", clf.classes_)
print("Number of classes:", len(clf.classes_))
print("Probability vector shape:", sample_probs.shape)

print("\n📊 Probabilities:")
for name, p in zip(accent_names, sample_probs):
    print(f"{name:10s} → {p*100:.2f}%")

print("\n🎉 Training completed successfully!")
