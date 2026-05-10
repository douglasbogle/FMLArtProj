import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read in data
df = pd.read_csv('phillips_data/final_cv_dataset.csv')
df["estimate_mid"] = (df["estimate_low"] + df["estimate_high"]) / 2

# Handle missing creation year
df["creation_year"] = df["creation_year"].fillna(df["creation_year"].median())

# Group rare artists into Other
top_artists = df["artist_name"].value_counts()
top_artists = top_artists[top_artists >= 5].index
df["artist_name"] = df["artist_name"].apply(lambda x: x if x in top_artists else "Other")
le = LabelEncoder()
df["artist_encoded"] = le.fit_transform(df["artist_name"])

# Normalize continuous features
df["surface_area_norm"] = (df["surface_area"] - df["surface_area"].mean()) / df["surface_area"].std()
df["creation_year_norm"] = (df["creation_year"] - df["creation_year"].mean()) / df["creation_year"].std()

class MetaDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        meta = torch.tensor([
            row["artist_encoded"],
            row["material"],
            row["surface_area_norm"],
            row["creation_year_norm"]
        ], dtype=torch.float32)
        label = torch.tensor(np.log1p(row["sold_price"]), dtype=torch.float32)
        return meta, label

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = MetaDataset(train_df)
val_dataset = MetaDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Baselines
mean_price = np.log1p(train_df["sold_price"]).mean()
baseline_mse = np.mean((np.log1p(val_df["sold_price"]) - mean_price) ** 2)
estimate_mse = np.mean((np.log1p(val_df["sold_price"]) - np.log1p(val_df["estimate_mid"])) ** 2)
print(f"Mean MSE: {baseline_mse:.4f}")
print(f"Estimate mid MSE: {estimate_mse:.4f}")

train_losses_all = []
val_losses_all = []
Path("results").mkdir(parents=True, exist_ok=True)

for epoch in range(30):
    model.train()
    train_losses = []

    for meta, labels in train_loader:
        meta, labels = meta.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(meta).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []

    with torch.no_grad():
        for meta, labels in val_loader:
            meta, labels = meta.to(device), labels.to(device)
            outputs = model(meta).squeeze()
            val_losses.append(criterion(outputs, labels).item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_losses_all.append(train_loss)
    val_losses_all.append(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "results/model_meta_only.pt")
np.save("results/train_losses_meta.npy", train_losses_all)
np.save("results/val_losses_meta.npy", val_losses_all)
np.save("results/baselines.npy", [baseline_mse, estimate_mse])