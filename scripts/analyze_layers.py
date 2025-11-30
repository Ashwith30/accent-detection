# scripts/analyze_layers.py
# ----------------------------------------------------
# HuBERT Layer-Wise Accent Classification Analysis
# ----------------------------------------------------
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=" * 70)
print("🎯 STEP 3: Analyzing HuBERT Layers for Accent Detection")
print("=" * 70)

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
labels = np.load("model/labels.npy", allow_pickle=True)
layer_accs = []

layers = list(range(13))
print(f"✅ Found {len(layers)} HuBERT layers (0–12)")
print(f"✅ Total samples: {len(labels)}\n")

# ----------------------------------------------------
# Define training function
# ----------------------------------------------------
def evaluate_layer(layer_idx):
    X = np.load(f"model/embeddings_layer_{layer_idx}.npy", allow_pickle=True)
    y = labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, y_test, preds

# ----------------------------------------------------
# Evaluate each layer
# ----------------------------------------------------
for layer in layers:
    print(f"\n🧩 Evaluating Layer {layer}...")
    acc, y_test, preds = evaluate_layer(layer)
    layer_accs.append(acc)
    print(f"✅ Accuracy (Layer {layer}): {acc:.4f}")

# ----------------------------------------------------
# Visualization
# ----------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(layers, layer_accs, marker='o')
plt.title("HuBERT Layer-wise Accuracy")
plt.xlabel("Layer Index")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(layers)
plt.savefig("model/layerwise_accuracy.png")
plt.show()

# ----------------------------------------------------
# Best layer
# ----------------------------------------------------
best_idx = np.argmax(layer_accs)
best_acc = layer_accs[best_idx]
print("\n🏆 BEST LAYER RESULTS 🏆")
print(f"Layer {best_idx} → Accuracy = {best_acc:.4f}")

# ----------------------------------------------------
# Detailed Report for Best Layer
# ----------------------------------------------------
print("\n📋 Classification Report (Best Layer):")
_, y_test, preds = evaluate_layer(best_idx)
print(classification_report(y_test, preds))

# ----------------------------------------------------
# Save results
# ----------------------------------------------------
np.save("model/layerwise_accuracy.npy", np.array(layer_accs))
print("\n💾 Results saved to model/layerwise_accuracy.npy")
print("📈 Graph saved to model/layerwise_accuracy.png")

print("=" * 70)
