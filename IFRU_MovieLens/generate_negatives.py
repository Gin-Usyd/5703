import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

np.random.seed(42)

# ===== Load data =====
data_dir = Path("./data/processed/")

train_df = pd.read_csv(data_dir / "train.csv")
valid_df = pd.read_csv(data_dir / "valid.csv")
test_df = pd.read_csv(data_dir / "test.csv")

# ===== Basic info =====
n_users = train_df["user_id"].max() + 1
n_items = train_df["item_id"].max() + 1

print(f"Users: {n_users}, Items: {n_items}")

# ===== Build user history =====
user_history = {}

for row in train_df.itertuples():
    user_history.setdefault(row.user_id, set()).add(row.item_id)

for row in valid_df.itertuples():
    user_history.setdefault(row.user_id, set()).add(row.item_id)

for row in test_df.itertuples():
    user_history.setdefault(row.user_id, set()).add(row.item_id)

# ===== Negative sampling =====
def sample_negatives(user_id, num_neg=100):
    negatives = []
    while len(negatives) < num_neg:
        item = np.random.randint(0, n_items)
        if item not in user_history[user_id]:
            negatives.append(item)
    return negatives

# ===== Generate =====
valid_neg = {}
test_neg = {}

print("Generating validation negatives...")
for row in tqdm(valid_df.itertuples()):
    valid_neg[row.user_id] = sample_negatives(row.user_id)

print("Generating test negatives...")
for row in tqdm(test_df.itertuples()):
    test_neg[row.user_id] = sample_negatives(row.user_id)

# ===== Save =====
out_dir = Path("./data/processed/")
with open(out_dir / "valid_neg.pkl", "wb") as f:
    pickle.dump(valid_neg, f)

with open(out_dir / "test_neg.pkl", "wb") as f:
    pickle.dump(test_neg, f)

print("Done!")

print(len(valid_neg), len(test_neg))