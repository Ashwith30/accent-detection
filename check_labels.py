import numpy as np
from collections import Counter

y = np.load("model/labels.npy", allow_pickle=True)
print("\nUnique accents:", set(y))
print("Counts per accent:", Counter(y))
print("Total labels:", len(y))
