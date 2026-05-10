import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Data setup
df = pd.read_csv('phillips_data/final_cv_dataset.csv')
df['image_path'] = df['image_path'].apply(lambda x: f'phillips_data/{x}')
df = df[df['image_path'].apply(lambda x: Path(x).exists())]

# Train test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Rebuild model
model = models.resnet50(weights=None)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("results/model_final.pt", map_location=device))

# Unfreeze layer4 so gradients can flow for Grad-CAM
for param in model.layer4.parameters():
    param.requires_grad = True

model.eval()
model.to(device)

# Target last conv layer
target_layer = [model.layer4[-1].conv3]
cam = GradCAM(model=model, target_layers=target_layer)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i in range(5):
    row = train_df.iloc[i]
    img = Image.open(row["image_path"]).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=img_tensor)

    img_display = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_display.astype(np.float32), grayscale_cam[0], use_rgb=True)

    axes[0, i].imshow(img.resize((224, 224)))
    with torch.no_grad():
        output = model(img_tensor)
        predicted_price = np.expm1(output.item())

    axes[0, i].set_title(f"Actual: ${row['sold_price']:,.0f}")
    axes[0, i].axis("off")

    axes[1, i].imshow(visualization)
    axes[1, i].set_title(f"Predicted: ${predicted_price:,.0f}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()