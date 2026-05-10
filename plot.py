import matplotlib.pyplot as plt
import numpy as np
import re

train_losses = []
val_losses = []

with open("results/.slurm/art_resnet-54857.out", "r") as f:
    for line in f:
        match = re.search(r"Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", line)
        if match:
            train_losses.append(float(match.group(1)))
            val_losses.append(float(match.group(2)))

# Convert to RMSE
train_losses = np.sqrt(train_losses)
val_losses = np.sqrt(val_losses)
baseline_mse = 3.1313
estimate_mse = 0.4127

epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, label="Train")
plt.plot(epochs, val_losses, label="Test")
plt.axhline(y=np.sqrt(baseline_mse), color='green', linestyle='--', label=f"Mean Baseline ({np.sqrt(baseline_mse):.2f})")
plt.axhline(y=np.sqrt(estimate_mse), color='red', linestyle='--', label=f"Estimate Mid Baseline ({np.sqrt(estimate_mse):.2f})")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Testing Loss (RMSE)")
plt.legend()
plt.show()
