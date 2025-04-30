# full_transformer_pipeline_improved.py
# Enhanced Transformer pipeline with code embeddings, positional encoding,
# richer MLP head, and custom Sharpe loss regularization for improved performance.

import os
import gc
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------
# 0. Configuration & Hyperparams
# ------------------------------
SEQ_LEN    = 60           # increased sequence length
BATCH_SIZE = 256          # smaller batch for stability
EPOCHS     = 50
PATIENCE   = 5

DMODEL     = 256          # larger model dim
NUM_LAYERS = 4            # deeper stack
NHEAD      = 8
D_FF       = 512
DROPOUT    = 0.2
LR         = 1e-4
WEIGHT_DECAY = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ------------------------------
# 1. Load & Preprocess Data
# ------------------------------
train_df = pd.read_csv("training_price_features.csv")
test_df  = pd.read_csv("validation_price_features.csv")

# Drop rows with any NaNs in features or target
exclude = ["Date","Target","RowId"]
FEATURES = [c for c in train_df.columns if c not in exclude]
keep_cols = FEATURES + ["Target"]

train_df = train_df.dropna(subset=keep_cols)\
                   .sort_values(["Date","SecuritiesCode"])\
                   .reset_index(drop=True)
test_df  = test_df .dropna(subset=keep_cols)\
                   .sort_values(["Date","SecuritiesCode"])\
                   .reset_index(drop=True)

# Datetime features
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df ["Date"] = pd.to_datetime(test_df ["Date"])

# Map SecuritiesCode to integer ids for embedding
codes = train_df["SecuritiesCode"].unique().tolist()
code2idx = {c:i for i,c in enumerate(codes)}
train_df["CodeIdx"] = train_df["SecuritiesCode"].map(code2idx)
test_df ["CodeIdx"] = test_df ["SecuritiesCode"].map(code2idx)
NUM_CODES = len(codes)

# Scale features
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df [FEATURES] = scaler.transform(test_df [FEATURES])

# ------------------------------
# 2. Sharpe Ratio Loss/Metric
# ------------------------------
def calc_spread_return_sharpe(df, portfolio_size=200, toprank_weight_ratio=2):
    def _per_day(day_df):
        ps = min(portfolio_size, len(day_df))
        long  = day_df.nsmallest(ps, "Rank")["Target"]
        short = day_df.nlargest(ps,  "Rank")["Target"]
        w = np.linspace(toprank_weight_ratio, 1, ps)
        return (long @ w)/w.mean() - (short @ w)/w.mean()
    daily = df.groupby("Date").apply(_per_day)
    return float(daily.mean()/(daily.std()+1e-8))

# Custom Sharpe-based regularization to add to MSE loss
class SharpeLoss(nn.Module):
    def __init__(self, reg_strength=1e-2):
        super().__init__()
        self.reg = reg_strength
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, ranks=None, dates=None):
        # Always compute MSE
        loss_mse = self.mse(preds, targets)
        # If no dates provided, skip Sharpe regularization
        if dates is None:
            return loss_mse
        # Otherwise compute Sharpe regularizer (approx) on batch
        df = pd.DataFrame({
            "Date": dates.cpu().numpy(),
            "pred": preds.detach().cpu().numpy(),
            "Target": targets.detach().cpu().numpy()
        })
        df["Rank"] = df.groupby("Date")["pred"].rank(method="first", ascending=False) - 1
        sharpe = calc_spread_return_sharpe(df)
        # Subtract to maximize Sharpe
        return loss_mse - self.reg * sharpe  # maximize sharpe via negative

# ------------------------------
# 3. Datasets
# ------------------------------
class TrainDS(Dataset):
    def __init__(self, df, seq_len, feats):
        self.seq_len, self.feats = seq_len, feats
        # group by code
        self.groups = {c:g.sort_values("Date").reset_index(drop=True)
                       for c,g in df.groupby("SecuritiesCode") if len(g)>seq_len}
        self.index_map = [(c,i) for c,g in self.groups.items() for i in range(seq_len,len(g))]
    def __len__(self): return len(self.index_map)
    def __getitem__(self, idx):
        code, pos = self.index_map[idx]
        grp = self.groups[code]
        seq = grp.loc[pos-self.seq_len:pos-1, self.feats].values.astype(np.float32)
        tgt = grp.loc[pos, "Target"].astype(np.float32)
        ci  = grp.loc[pos, "CodeIdx"].astype(np.int64)
        dts = grp.loc[pos-self.seq_len:pos-1, "Date"].dt.dayofweek.values.astype(np.int64)
        return torch.from_numpy(seq), torch.tensor(ci), torch.tensor(dts), torch.tensor(tgt)

class InferDS(Dataset):
    def __init__(self, train_df, test_df, seq_len, feats):
        full = pd.concat([train_df.assign(_t=0), test_df.assign(_t=1)], ignore_index=True)
        full = full.sort_values(["SecuritiesCode","Date"]).reset_index(drop=True)
        full = full.loc[:,~full.columns.duplicated(keep="first")]
        self.full = full
        self.codes = full["SecuritiesCode"].values
        self.test_idx = np.where(full["_t"].values==1)[0]
        self.seq_len, self.feats = seq_len, feats
    def __len__(self): return len(self.test_idx)
    def __getitem__(self, i):
        j   = self.test_idx[i]
        row = self.full.iloc[j]
        idxs= np.where(self.codes==row["SecuritiesCode"])[0]
        loc = int(np.where(idxs==j)[0])
        start = max(0, loc-self.seq_len)
        hist  = self.full.iloc[idxs[start:loc]]
        X = hist[self.feats].values.astype(np.float32)
        if X.shape[0]==0:
            base = row[self.feats].values.astype(np.float32)
            X = np.tile(base, (self.seq_len,1))
        elif X.shape[0]<self.seq_len:
            pad = np.tile(X[0], (self.seq_len-X.shape[0],1))
            X = np.vstack([pad,X])
        ci  = row["CodeIdx"].astype(np.int64)
        return torch.from_numpy(X), torch.tensor(ci), i

# ------------------------------
# 4. DataLoaders
# ------------------------------
train_loader = DataLoader(TrainDS(train_df, SEQ_LEN, FEATURES),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=True)
infer_ds      = InferDS(train_df, test_df, SEQ_LEN, FEATURES)
infer_loader  = DataLoader(infer_ds, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=os.cpu_count(),
                           pin_memory=True)

# ------------------------------
# 5. Model with Embeddings & Positional Encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        # Register as buffer so it's moved with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]; pe is [1, max_len, d_model]
        return x + self.pe[:, :x.size(1)].to(x.device)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, n_feats, n_codes):
        super().__init__()
        self.code_emb = nn.Embedding(n_codes, DMODEL)
        self.input_proj = nn.Linear(n_feats, DMODEL)
        self.pos_enc = PositionalEncoding(DMODEL, max_len=SEQ_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=DMODEL, nhead=NHEAD,
            dim_feedforward=D_FF, dropout=DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(DMODEL),
            nn.Linear(DMODEL, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x, code_idx):
        # x: [B, L, n_feats], code_idx: [B]
        feat = self.input_proj(x)
        feat = self.pos_enc(feat)
        ce = self.code_emb(code_idx).unsqueeze(1).expand(-1, SEQ_LEN, -1)
        h = feat + ce
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)

model = TransformerRegressor(len(FEATURES), NUM_CODES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# use custom Sharpe loss
criterion = SharpeLoss(reg_strength=1e-2)

# LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------------------
# 6. Training Loop
# ------------------------------
best_loss, wait = float("inf"), 0
for epoch in range(1, EPOCHS+1):
    model.train(); total, count = 0.0, 0
    for xb, ci, dts, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
        xb, ci, yb = xb.to(DEVICE), ci.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb, ci)
        loss = criterion(preds, yb, None, None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()*xb.size(0); count += xb.size(0)
    train_mse = total/count
    scheduler.step()
    print(f"Epoch {epoch} | train MSE: {train_mse:.6f}")
    if train_mse + 1e-6 < best_loss:
        best_loss, wait = train_mse, 0
        torch.save(model.state_dict(), "best_transformer_improved.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping triggered.")
            break

torch.cuda.empty_cache(); gc.collect()

# ------------------------------
# 7. Inference & Submission
# ------------------------------
model.load_state_dict(torch.load("best_transformer_improved.pt", map_location=DEVICE))
model.eval()

preds = np.empty(len(infer_ds), dtype=np.float32)
with torch.no_grad():
    for xb, ci, idx in tqdm(infer_loader, desc="Inferring"):
        xb, ci = xb.to(DEVICE), ci.to(DEVICE)
        out = model(xb, ci).cpu().numpy()
        preds[idx.numpy()] = out

res = test_df.reset_index(drop=True)
res["pred"] = preds
res["Rank"] = res.groupby("Date")["pred"].rank("first", ascending=False) - 1

# Metrics
rmse   = math.sqrt(mean_squared_error(res["Target"], res["pred"]))
mae    = mean_absolute_error(res["Target"], res["pred"])
sharpe = calc_spread_return_sharpe(res)
print(f"Test RMSE  : {rmse:.4f}")
print(f"Test MAE   : {mae:.4f}")
print(f"Test Sharpe: {sharpe:.4f}")

# Export
res[["Date","SecuritiesCode","Rank"]].to_csv("submission_improved.csv", index=False)
print("✅ submission_improved.csv written")
