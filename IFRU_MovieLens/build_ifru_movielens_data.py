import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =========================
# Config
# =========================
seed = 42
unlearn_ratio = 0.02

# 你自己处理好的数据目录
src_dir = Path("./data/processed")

# 作者仓库需要的数据目录
dst_dir = Path("./data/IFRU/Data/Process/MovieLens/0.02")
dst_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(seed)

# =========================
# Load your processed data
# =========================
train_df = pd.read_csv(src_dir / "train.csv")   # columns: user_id, item_id
valid_df = pd.read_csv(src_dir / "valid.csv")
test_df = pd.read_csv(src_dir / "test.csv")

with open(src_dir / "valid_neg.pkl", "rb") as f:
    valid_neg = pickle.load(f)

with open(src_dir / "test_neg.pkl", "rb") as f:
    test_neg = pickle.load(f)

# =========================
# Step 1: split train into normal/random
# =========================
n_random = int(len(train_df) * unlearn_ratio)
random_idx = np.random.choice(train_df.index, size=n_random, replace=False)

train_random = train_df.loc[random_idx].copy()
train_normal = train_df.drop(random_idx).copy()

train_random["label"] = 1
train_normal["label"] = 1

train_random = train_random.rename(columns={"user_id": "user", "item_id": "item"})
train_normal = train_normal.rename(columns={"user_id": "user", "item_id": "item"})

# =========================
# Step 2: build valid with 1 positive + negatives
# =========================
valid_rows = []
for row in valid_df.itertuples(index=False):
    u = int(row.user_id)
    pos_i = int(row.item_id)

    valid_rows.append([u, pos_i, 1])

    for neg_i in valid_neg[u]:
        valid_rows.append([u, int(neg_i), 0])

valid_out = pd.DataFrame(valid_rows, columns=["user", "item", "label"])

# =========================
# Step 3: build test with 1 positive + negatives
# =========================
test_rows = []
for row in test_df.itertuples(index=False):
    u = int(row.user_id)
    pos_i = int(row.item_id)

    test_rows.append([u, pos_i, 1])

    for neg_i in test_neg[u]:
        test_rows.append([u, int(neg_i), 0])

test_out = pd.DataFrame(test_rows, columns=["user", "item", "label"])

# =========================
# Save
# =========================
train_normal.to_csv(dst_dir / "train_normal.csv", index=False)
train_random.to_csv(dst_dir / "train_random.csv", index=False)
valid_out.to_csv(dst_dir / "valid.csv", index=False)
test_out.to_csv(dst_dir / "test.csv", index=False)

print("===== Done =====")
print(f"train_normal: {train_normal.shape}")
print(f"train_random: {train_random.shape}")
print(f"valid: {valid_out.shape}")
print(f"test: {test_out.shape}")
print(f"saved to: {dst_dir}")

print(valid_out["label"].value_counts())
print(test_out["label"].value_counts())