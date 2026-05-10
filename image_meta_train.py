import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read in data
df = pd.read_csv('phillips_data/final_cv_dataset.csv')
df["estimate_mid"] = (df["estimate_low"] + df["estimate_high"]) / 2
df['image_path'] = df['image_path'].apply(lambda x: f'phillips_data/{x}')
df = df[df['image_path'].apply(lambda x: Path(x).exists())]

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

NUM_META_FEATURES = 4  # artist_encoded, material, surface_area_norm, creation_year_norm

class ArtDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        meta = torch.tensor([
            row["artist_encoded"],
            row["material"],
            row["surface_area_norm"],
            row["creation_year_norm"]
        ], dtype=torch.float32)
        label = torch.tensor(np.log1p(row["sold_price"]), dtype=torch.float32)
        return image, meta, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ArtDataset(train_df, transform=transform)
val_dataset = ArtDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

class ArtModelWithMeta(nn.Module):
    def __init__(self, num_meta_features):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        for param in backbone.parameters():
            param.requires_grad = False
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048 + num_meta_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, meta):
        img_features = self.image_features(image).squeeze(-1).squeeze(-1)
        combined = torch.cat([img_features, meta], dim=1)
        return self.head(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArtModelWithMeta(NUM_META_FEATURES).to(device)

optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
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

for epoch in range(150):
    model.train()
    train_losses = []

    for images, meta, labels in train_loader:
        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, meta).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []

    with torch.no_grad():
        for images, meta, labels in val_loader:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            outputs = model(images, meta).squeeze()
            val_losses.append(criterion(outputs, labels).item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_losses_all.append(train_loss)
    val_losses_all.append(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "results/model_final2.pt")
np.save("results/train_losses.npy", train_losses_all)
np.save("results/val_losses.npy", val_losses_all)
np.save("results/baselines.npy", [baseline_mse, estimate_mse])