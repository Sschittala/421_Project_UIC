# scripts/train_ann.py
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, classification_report
from models.ann_model import ANNModel

# ---- config (paths resolved relative to this file) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../P1_DATA")
OUT_DIR  = os.path.join(BASE_DIR, "../outputs")
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
INPUT_DIM = 384
NUM_CLASSES = 3

os.makedirs(OUT_DIR, exist_ok=True)

# ---- reproducibility ----
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")  # Q1 runs on CPU

# ---- tolerant CSV reader (must match embeddings.py) ----
def read_split(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"trac2_CONVT_{split}.csv")
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip"
    )

# ---- load dataframes ----
train_df = read_split("train")
dev_df   = read_split("dev")
test_df  = read_split("test")

COL_EMO = "Emotion"
COL_POL = "EmotionalPolarity"
COL_EMP = "Empathy"

# Robust coercion: handle strings like "positive", blanks, and float-y values
def coerce_polarity_inplace(df, col):
    # map common string labels to numbers
    mapper = {
        "negative": 0, "neg": 0, "-1": 0,
        "neutral":  1, "neu": 1,  "0": 1,
        "positive": 2, "pos": 2,  "1": 2
    }
    # start from original col as string
    raw = df[col].astype(str).str.strip().str.lower()
    mapped = raw.map(mapper)

    # numeric fallback (handles "2", "2.0", etc.)
    numeric = pd.to_numeric(df[col], errors="coerce")

    # prefer numeric if available, else mapped
    s = numeric
    s = s.where(~s.isna(), mapped)

    # round to nearest, fill bads with -1, cast to plain int
    s = pd.to_numeric(s, errors="coerce").round().fillna(-1).astype(int)

    # write back
    df[col] = s

coerce_polarity_inplace(train_df, COL_POL)
coerce_polarity_inplace(dev_df,   COL_POL)
# test has no labels, so no need to coerce

# ---- load embeddings ----
X_train = torch.tensor(np.load(os.path.join(OUT_DIR, "train_embeddings.npy")), dtype=torch.float32)
X_dev   = torch.tensor(np.load(os.path.join(OUT_DIR, "dev_embeddings.npy")),   dtype=torch.float32)
X_test  = torch.tensor(np.load(os.path.join(OUT_DIR, "test_embeddings.npy")),  dtype=torch.float32)

# sanity: shapes must match the frames we will keep
assert X_train.shape[0] == len(train_df), f"train rows {len(train_df)} != train emb {X_train.shape[0]}"
assert X_dev.shape[0]   == len(dev_df),   f"dev rows {len(dev_df)} != dev emb {X_dev.shape[0]}"
assert X_test.shape[0]  == len(test_df),  f"test rows {len(test_df)} != test emb {X_test.shape[0]}"

# ---- drop invalid polarity rows and keep alignment with embeddings
valid_train_mask = train_df[COL_POL].isin([0, 1, 2])
if not valid_train_mask.all():
    X_train = X_train[torch.tensor(valid_train_mask.values, dtype=torch.bool)]
    train_df = train_df.loc[valid_train_mask].reset_index(drop=True)

valid_dev_mask = dev_df[COL_POL].isin([0, 1, 2])
if not valid_dev_mask.all():
    X_dev = X_dev[torch.tensor(valid_dev_mask.values, dtype=torch.bool)]
    dev_df = dev_df.loc[valid_dev_mask].reset_index(drop=True)

# ---- targets
y_train_emotion  = torch.tensor(train_df[COL_EMO].values, dtype=torch.float32).unsqueeze(1)
y_train_empathy  = torch.tensor(train_df[COL_EMP].values, dtype=torch.float32).unsqueeze(1)
y_train_polarity = torch.tensor(train_df[COL_POL].values, dtype=torch.long)

y_dev_emotion  = torch.tensor(dev_df[COL_EMO].values, dtype=torch.float32).unsqueeze(1)
y_dev_empathy  = torch.tensor(dev_df[COL_EMP].values, dtype=torch.float32).unsqueeze(1)
y_dev_polarity = torch.tensor(dev_df[COL_POL].values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train_emotion, y_train_polarity, y_train_empathy)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---- model/loss/optimizer ----
model = ANNModel(input_dim=INPUT_DIM, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
criterion_reg = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- training ----
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, y_e, y_p, y_emp in train_dl:
        xb   = xb.to(device)
        y_e  = y_e.to(device)
        y_p  = y_p.to(device)
        y_emp= y_emp.to(device)

        optimizer.zero_grad()
        pred_e, pred_p, pred_emp = model(xb)
        loss = criterion_reg(pred_e, y_e) + criterion_cls(pred_p, y_p) + criterion_reg(pred_emp, y_emp)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(train_dl):.4f}")

# ---- evaluation on dev ----
model.eval()
with torch.no_grad():
    dev_e, dev_p, dev_emp = model(X_dev.to(device))
    mae_e   = mean_absolute_error(dev_df[COL_EMO], dev_e.cpu().squeeze())
    mae_emp = mean_absolute_error(dev_df[COL_EMP], dev_emp.cpu().squeeze())
    preds_p = dev_p.argmax(dim=1).cpu()
    print("Dev Emotion MAE:", mae_e)
    print("Dev Empathy MAE:", mae_emp)
    print("Dev Polarity report:\n", classification_report(dev_df[COL_POL], preds_p))

# ---- predict test and export CSV ----
with torch.no_grad():
    te_e, te_p, te_emp = model(X_test.to(device))
    te_e   = te_e.cpu().squeeze().numpy()
    te_emp = te_emp.cpu().squeeze().numpy()
    te_pol = te_p.argmax(dim=1).cpu().numpy()

# Use provided id column if present, else synthesize
if "id" in test_df.columns:
    ids = test_df["id"].values
else:
    ids = np.arange(len(test_df))

pred_df = pd.DataFrame({
    "id": ids,
    "Emotion": te_e,
    "EmotionalPolarity": te_pol,
    "Empathy": te_emp
})

out_csv = os.path.join(OUT_DIR, "predictions_ann.csv")
pred_df.to_csv(out_csv, index=False)
print(f"Saved test predictions to: {out_csv}")
