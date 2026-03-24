import pandas as pd
from pathlib import Path

# 1. 原始数据路径
raw_path = Path("./data/ratings.dat")

# 2. 读取 ratings.dat
df = pd.read_csv(
    raw_path,
    sep="::",
    engine="python",
    names=["user_raw", "item_raw", "rating", "timestamp"]
)

# 3. 打印基础信息
print("===== Basic Info =====")
print(df.head())
print(f"Total interactions: {len(df)}")
print(f"Total users: {df['user_raw'].nunique()}")
print(f"Total items: {df['item_raw'].nunique()}")

print("\n===== Rating Distribution =====")
print(df["rating"].value_counts().sort_index())

print("\n===== User Interaction Stats =====")
user_counts = df.groupby("user_raw").size()
print(user_counts.describe())

# ===== Step 1: Convert to implicit feedback =====
df_pos = df[df["rating"] >= 4].copy()

print("\n===== After Implicit Conversion =====")
print(f"Interactions: {len(df_pos)}")
print(f"Users: {df_pos['user_raw'].nunique()}")
print(f"Items: {df_pos['item_raw'].nunique()}")


# ===== Step 2: Filter users with >=3 interactions =====
user_counts = df_pos.groupby("user_raw").size()
valid_users = user_counts[user_counts >= 3].index

df_pos = df_pos[df_pos["user_raw"].isin(valid_users)]

print("\n===== After User Filtering =====")
print(f"Users: {df_pos['user_raw'].nunique()}")

# ===== Step 3: Re-index users and items =====
import json
from pathlib import Path

user2id = {int(u): int(i) for i, u in enumerate(df_pos["user_raw"].unique())}
item2id = {int(it): int(j) for j, it in enumerate(df_pos["item_raw"].unique())}

df_pos["user_id"] = df_pos["user_raw"].map(user2id)
df_pos["item_id"] = df_pos["item_raw"].map(item2id)

print("\n===== After Reindex =====")
print(f"Users: {df_pos['user_id'].nunique()}")
print(f"Items: {df_pos['item_id'].nunique()}")
print(f"user_id range: {df_pos['user_id'].min()} ~ {df_pos['user_id'].max()}")
print(f"item_id range: {df_pos['item_id'].min()} ~ {df_pos['item_id'].max()}")

out_dir = Path("./data/processed/")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "user2id.json", "w") as f:
    json.dump(user2id, f)

with open(out_dir / "item2id.json", "w") as f:
    json.dump(item2id, f)

# ===== Step 4: Time-based split =====
df_pos = df_pos.sort_values(["user_id", "timestamp"])

train_list = []
valid_list = []
test_list = []

for user_id, group in df_pos.groupby("user_id"):
    items = group["item_id"].tolist()

    if len(items) < 3:
        continue

    train_items = items[:-2]
    valid_item = items[-2]
    test_item = items[-1]

    for i in train_items:
        train_list.append([user_id, i])

    valid_list.append([user_id, valid_item])
    test_list.append([user_id, test_item])

import pandas as pd

train_df = pd.DataFrame(train_list, columns=["user_id", "item_id"])
valid_df = pd.DataFrame(valid_list, columns=["user_id", "item_id"])
test_df = pd.DataFrame(test_list, columns=["user_id", "item_id"])

train_df.to_csv(out_dir / "train.csv", index=False)
valid_df.to_csv(out_dir / "valid.csv", index=False)
test_df.to_csv(out_dir / "test.csv", index=False)

print("\n===== Split Done =====")
print(f"Train: {len(train_df)}")
print(f"Valid: {len(valid_df)}")
print(f"Test: {len(test_df)}")