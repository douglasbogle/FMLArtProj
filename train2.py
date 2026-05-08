import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt

# --- 1. DATA PREP ---
def prepare_data():
    df = pd.read_csv('phillips_data/final_cv_dataset.csv')
    df = df.dropna(subset=['sold_price', 'estimate_low', 'estimate_high'])
    df = df[(df['sold_price'] > 0) & (df['estimate_low'] > 0)].copy()
    
    df['image_path'] = df['image_path'].apply(lambda x: f'phillips_data/contemporary_images/{Path(x).name}')
    df = df[df['image_path'].apply(lambda x : Path(x).exists())].reset_index(drop=True)
    
    df["estimate_mid"] = (df["estimate_low"] + df["estimate_high"]) / 2
    
    cat_cols = ['material', 'surface'] 
    num_cols = ['estimate_low', 'estimate_high', 'surface_area', 'auction_year', 'creation_year']
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    df[num_cols] = StandardScaler().fit_transform(df[num_cols].fillna(0))
    
    return df, cat_cols, num_cols

# --- 2. VECTOR DATASET ---
class FeatureDataset(Dataset):
    def __init__(self, df_subset, features_subset, cat_cols, num_cols):
        self.df = df_subset.reset_index(drop=True)
        self.features = features_subset
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        visual_feat = self.features[idx]
        tab_feat = torch.tensor(row[self.cat_cols + self.num_cols].values.astype(np.float32))
        
        actual_log = np.log(max(float(row["sold_price"]), 1.0))
        est_log = np.log(max(float(row["estimate_mid"]), 1.0))
        
        target = torch.tensor(actual_log - est_log, dtype=torch.float32)
        return visual_feat, tab_feat, target, torch.tensor(est_log, dtype=torch.float32)

# --- 3. SLIM ARTNET ARCHITECTURE ---
class ArtNetResidual(nn.Module):
    def __init__(self, num_tabular_features):
        super().__init__()
        self.visual_bottleneck = nn.Sequential(
            nn.Linear(2048, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 16), 
            nn.ReLU()
        )
        self.tab_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 32), 
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16), 
            nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(16 + 16, 32), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32, 1)
        )

    def forward(self, vis, tab, est_log, training=True):
        if training:
            vis = vis + torch.randn_like(vis) * 0.01 
        v_f = self.visual_bottleneck(vis)
        t_f = self.tab_branch(tab)
        residual_pred = self.fusion_head(torch.cat((v_f, t_f), dim=1)).squeeze(-1)
        return residual_pred + est_log, residual_pred

# --- 4. PREDICTION FUNCTION ---
def predict_alpha(model, vis_tensor, tab_tensor, est_mid_val, device):
    model.eval()
    with torch.no_grad():
        vis = vis_tensor.to(device).unsqueeze(0) if vis_tensor.ndim == 1 else vis_tensor.to(device)
        tab = tab_tensor.to(device).unsqueeze(0) if tab_tensor.ndim == 1 else tab_tensor.to(device)
        est_log = torch.tensor([np.log(est_mid_val)], dtype=torch.float32).to(device)
        final_log_price, alpha = model(vis, tab, est_log, training=False)
        return np.exp(final_log_price.item()), alpha.item()

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    df, cat_cols, num_cols = prepare_data()
    all_visual_features = torch.load('phillips_data/image_features.pt')
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_df = df.iloc[val_idx]
    
    # Expert Baseline Calculation
    val_log_actual = np.log(val_df["sold_price"].clip(lower=1.0))
    val_log_est_mid = np.log(val_df["estimate_mid"].clip(lower=1.0))
    est_val_rmse = np.sqrt(np.mean((val_log_actual - val_log_est_mid)**2))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ArtNetResidual(len(cat_cols + num_cols)).to(device)
    
    MODEL_PATH = 'best_slim_artnet.pt'
    train_history, val_history = [], []

    # Check for existing weights
    if os.path.exists(MODEL_PATH):
        print(f"--- Loading Saved Model: {MODEL_PATH} ---")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"--- No Model Found. Starting Training Loop ---")
        train_loader = DataLoader(FeatureDataset(df.iloc[train_idx], all_visual_features[train_idx], cat_cols, num_cols), batch_size=256, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_df, all_visual_features[val_idx], cat_cols, num_cols), batch_size=256)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.MSELoss()

        num_epochs, best_val_rmse, early_stop_counter = 100, float('inf'), 0
        print(f"Expert Baseline RMSE: {est_val_rmse:.4f}")

        for epoch in range(num_epochs):
            model.train()
            t_losses = []
            for vis, tab, targets, est_logs in train_loader:
                vis, tab, targets, est_logs = vis.to(device), tab.to(device), targets.to(device), est_logs.to(device)
                optimizer.zero_grad()
                final_pred, _ = model(vis, tab, est_logs, training=True)
                loss = criterion(final_pred, targets + est_logs)
                loss.backward(); optimizer.step()
                t_losses.append(loss.item())
            
            model.eval()
            v_losses = []
            with torch.no_grad():
                for vis, tab, targets, est_logs in val_loader:
                    vis, tab, targets, est_logs = vis.to(device), tab.to(device), targets.to(device), est_logs.to(device)
                    final_pred, _ = model(vis, tab, est_logs, training=False)
                    v_losses.append(criterion(final_pred, targets + est_logs).item())
            
            train_rmse, val_rmse = np.sqrt(np.mean(t_losses)), np.sqrt(np.mean(v_losses))
            train_history.append(train_rmse); val_history.append(val_rmse)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), MODEL_PATH)
                early_stop_counter = 0
            else: early_stop_counter += 1
            
            scheduler.step(val_rmse)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")
            if early_stop_counter >= 12: 
                print(f"Early stopping at epoch {epoch+1}.")
                break

    # --- 6. UNIFIED VISUALIZATION ---
    # Re-run val_loader for scatter plot even if loaded from disk
    val_loader = DataLoader(FeatureDataset(val_df, all_visual_features[val_idx], cat_cols, num_cols), batch_size=256)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for vis, tab, targets, est_logs in val_loader:
            vis, tab, targets, est_logs = vis.to(device), tab.to(device), targets.to(device), est_logs.to(device)
            final_pred, _ = model(vis, tab, est_logs, training=False)
            preds.extend(final_pred.cpu().numpy()); actuals.extend((targets + est_logs).cpu().numpy())

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    if train_history: # Only plot curves if we actually trained
        plt.plot(train_history, label='Train RMSE', alpha=0.7)
        plt.plot(val_history, label='Val RMSE', color='darkblue', lw=2)
    plt.axhline(y=est_val_rmse, color='red', ls='--', label=f'Expert Baseline ({est_val_rmse:.3f})')
    plt.title('Training Convergence')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(actuals, preds, alpha=0.2, color='green', s=15)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', ls='--')
    plt.title('Predicted vs Actual (Log Price)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 7. FINDING A "WINNING" LOT ---
    print("\n--- Hunting for a 'Win' (ArtNet Error < Expert Error) ---")
    orig_df = pd.read_csv('phillips_data/final_cv_dataset.csv')
    
    for i in range(len(val_idx)):
        v_idx = val_idx[i]
        raw_row = orig_df.iloc[v_idx]
        raw_mid = (raw_row['estimate_low'] + raw_row['estimate_high']) / 2
        raw_sold = raw_row['sold_price']
        
        test_tab = torch.tensor(df.iloc[val_idx[i]][cat_cols + num_cols].values.astype(np.float32))
        pred_p, alpha = predict_alpha(model, all_visual_features[v_idx], test_tab, raw_mid, device)
        
        expert_err = abs(raw_sold - raw_mid)
        model_err = abs(raw_sold - pred_p)
        
        if model_err < expert_err:
            print(f"SUCCESS at Lot Index {v_idx}")
            print(f"Expert Mid-Estimate: ${raw_mid:,.2f}")
            print(f"Actual Sold Price:   ${raw_sold:,.2f}")
            print("-" * 30)
            print(f"ArtNet Prediction:   ${pred_p:,.2f} (Alpha: {alpha:+.2%})")
            print(f"Expert Error: ${expert_err:,.2f} | ArtNet Error: ${model_err:,.2f}")
            break