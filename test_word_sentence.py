from test_audio import extract_embedding, classifier, scaler, ACCENTS
import numpy as np

def predict_from_embedding(emb):
    emb_scaled = scaler.transform(emb)
    probs = classifier.predict_proba(emb_scaled)[0]
    idx = int(np.argmax(probs))
    accent = ACCENTS[idx]
    return accent, probs

# -------------------------------
# Test both word + sentence audio
# -------------------------------
def compare(word_path, sentence_path):
    print("\n🔍 Testing WORD-level audio")
    emb_word = extract_embedding(word_path)
    acc_word, probs_word = predict_from_embedding(emb_word)

    print("\n🔍 Testing SENTENCE-level audio")
    emb_sent = extract_embedding(sentence_path)
    acc_sent, probs_sent = predict_from_embedding(emb_sent)

    print("\n========== RESULT ==========")
    print(f"🎤 Word-level predicted: {acc_word}")
    print(f"🗣 Sentence-level predicted: {acc_sent}")

    print("\n📊 Word-level probabilities:")
    for a, p in zip(ACCENTS, probs_word):
        print(f"{a:10s} → {p*100:.2f}%")

    print("\n📊 Sentence-level probabilities:")
    for a, p in zip(ACCENTS, probs_sent):
        print(f"{a:10s} → {p*100:.2f}%")

# -------------------------------
# RUN TEST
# -------------------------------
if __name__ == "__main__":
    word = input("Word audio path: ")
    sentence = input("Sentence audio path: ")
    compare(word, sentence)
