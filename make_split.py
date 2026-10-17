import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = 0.10

df = pd.read_csv("data/processed/manifest.csv")

# 1) Test'te olup train'de de bulunan hash'leri test'ten çıkar
train_hashes = set(df.loc[df["split"] == "train", "hash"])
leak = (df["split"] == "test") & (df["hash"].isin(train_hashes))
df = df[~leak]

# 2) Her hash için tek kopya bırak (test ve train içi tekrarlar)
df = df.drop_duplicates(subset="hash", keep="first")

# 3) Train'in %10'unu sabit seed ile val yap
train_df = df[df["split"] == "train"]
_, val_idx = train_test_split(
    train_df.index,
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=train_df["label"],
)
df.loc[val_idx, "split"] = "val"

df.to_csv("data/processed/manifest_split.csv", index=False)
print(df.groupby(["split", "label"]).size())
print("toplam:", len(df), "| tekrar eden hash:", df["hash"].duplicated().sum())