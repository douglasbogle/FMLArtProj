import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Read in data, add in the middle of estimated high and lows
df = pd.read_csv('phillips_data/final_cv_dataset.csv')
df["estimate_mid"] = (df["estimate_low"] + df["estimate_high"]) / 2

# Adjust image path
df['image_path'] = df['image_path'].apply(lambda x: f'phillips_data/{x}')

# Filter out df for only entries we have images for
df = df[df['image_path'].apply(lambda x : Path(x).exists())]

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
        label = torch.tensor(np.log1p(row["sold_price"]), dtype=torch.float32)
        return image, label

# For scaling image sizes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ArtDataset(train_df, transform=transform)
val_dataset = ArtDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet50(weights="IMAGENET1K_V1")

# Use mean as a baseline
mean_price = np.log1p(train_df["sold_price"]).mean()
baseline_mse = np.mean((np.log1p(val_df["sold_price"]) - mean_price) ** 2)
print(f"Mean MSE: {baseline_mse:.4f}")

# Use mid of estimates as another baseline
estimate_mse = np.mean((np.log1p(val_df["sold_price"]) - np.log1p(val_df["estimate_mid"])) ** 2)
print(f"Estimate mid MSE: {estimate_mse:.4f}")

# Freeze all layers of ResNet
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer with our own regression head to make price predictions
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(30):
    # Training
    model.train()
    train_losses = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Testing
    model.eval()
    val_losses = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            val_losses.append(criterion(outputs, labels).item())

    print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")
    torch.save(model.state_dict(), f"results/model_epoch{epoch+1}.pt")