import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_csv('phillips_data/final_cv_dataset.csv')
df['image_path'] = df['image_path'].apply(lambda x: f'phillips_data/contemporary_images/{Path(x).name}')
df = df[df['image_path'].apply(lambda x : Path(x).exists())].reset_index(drop=True)

# Pre-trained ResNet (Frozen)
backbone = models.resnet50(weights="IMAGENET1K_V1").to(device)
backbone.fc = nn.Identity()
backbone.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

features = []
print("Extracting features (One-time cost)...")
with torch.no_grad():
    for path in tqdm(df['image_path']):
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            feat = backbone(img).cpu().squeeze()
            features.append(feat)
        except:
            features.append(torch.zeros(2048)) # Placeholder for broken images

# Save as a single tensor
torch.save(torch.stack(features), 'phillips_data/image_features.pt')
print("Done! You now have a high-speed feature file.")