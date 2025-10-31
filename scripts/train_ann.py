# scripts/train_ann.py
"""
Train ANN model for emotion, polarity, and empathy prediction.
Supports both GloVe and Sentence-BERT embeddings.
"""
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
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
from models.ann_model import ANNModel

# ---- CONFIG ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../P1_DATA")
OUT_DIR  = os.path.join(BASE_DIR, "../outputs")

# Choose embedding type: "sbert" or "glove"
EMBEDDING_TYPE = "sbert"  # Change to "glove" to use GloVe embeddings

BATCH_SIZE = 32
EPOCHS = 50  # Increased for better convergence
LR = 1e-3
NUM_CLASSES = 3

# Set input dimension based on embedding type
if EMBEDDING_TYPE == "sbert":
    INPUT_DIM = 384  # Sentence-BERT dimension
    EMB_SUFFIX = "sbert"
elif EMBEDDING_TYPE == "glove":
    INPUT_DIM = 100  # GloVe dimension
    EMB_SUFFIX = "glove"
else:
    raise ValueError(f"Unknown embedding type: {EMBEDDING_TYPE}")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Reproducibility ----
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")  # Q1 runs on CPU

# ---- CSV Reader ----
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

# ---- Load Dataframes ----
print(f"\n{'='*60}")
print(f"TRAINING ANN WITH {EMBEDDING_TYPE.upper()} EMBEDDINGS")
print(f"{'='*60}\n")

train_df = read_split("train")
dev_df   = read_split("dev")
test_df  = read_split("test")

COL_EMO = "Emotion"
COL_POL = "EmotionalPolarity"
COL_EMP = "Empathy"

# ---- Polarity Coercion ----
def coerce_polarity_inplace(df, col):
    mapper = {
        "negative": 0, "neg": 0, "-1": 0,
        "neutral":  1, "neu": 1,  "0": 1,
        "positive": 2, "pos": 2,  "1": 2
    }
    raw = df[col].astype(str).str.strip().str.lower()
    mapped = raw.map(mapper)
    numeric = pd.to_numeric(df[col], errors="coerce")
    s = numeric.where(~numeric.isna(), mapped)
    s = pd.to_numeric(s, errors="coerce").round().fillna(-1).astype(int)
    df[col] = s

coerce_polarity_inplace(train_df, COL_POL)
coerce_polarity_inplace(dev_df, COL_POL)

# ---- Load Embeddings ----
X_train = torch.tensor(np.load(os.path.join(OUT_DIR, f"train_embeddings_{EMB_SUFFIX}.npy")), dtype=torch.float32)
X_dev   = torch.tensor(np.load(os.path.join(OUT_DIR, f"dev_embeddings_{EMB_SUFFIX}.npy")),   dtype=torch.float32)
X_test  = torch.tensor(np.load(os.path.join(OUT_DIR, f"test_embeddings_{EMB_SUFFIX}.npy")),  dtype=torch.float32)

print(f"Loaded embeddings:")
print(f"  Train: {X_train.shape}")
print(f"  Dev:   {X_dev.shape}")
print(f"  Test:  {X_test.shape}\n")

# Sanity checks
assert X_train.shape[0] == len(train_df)
assert X_dev.shape[0] == len(dev_df)
assert X_test.shape[0] == len(test_df)

# ---- Drop Invalid Polarity Rows ----
valid_train_mask = train_df[COL_POL].isin([0, 1, 2])
if not valid_train_mask.all():
    X_train = X_train[torch.tensor(valid_train_mask.values, dtype=torch.bool)]
    train_df = train_df.loc[valid_train_mask].reset_index(drop=True)

valid_dev_mask = dev_df[COL_POL].isin([0, 1, 2])
if not valid_dev_mask.all():
    X_dev = X_dev[torch.tensor(valid_dev_mask.values, dtype=torch.bool)]
    dev_df = dev_df.loc[valid_dev_mask].reset_index(drop=True)

# ---- Prepare Targets ----
y_train_emotion  = torch.tensor(train_df[COL_EMO].values, dtype=torch.float32).unsqueeze(1)
y_train_empathy  = torch.tensor(train_df[COL_EMP].values, dtype=torch.float32).unsqueeze(1)
y_train_polarity = torch.tensor(train_df[COL_POL].values, dtype=torch.long)

y_dev_emotion  = torch.tensor(dev_df[COL_EMO].values, dtype=torch.float32).unsqueeze(1)
y_dev_empathy  = torch.tensor(dev_df[COL_EMP].values, dtype=torch.float32).unsqueeze(1)
y_dev_polarity = torch.tensor(dev_df[COL_POL].values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train_emotion, y_train_polarity, y_train_empathy)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model, Loss, Optimizer ----
model = ANNModel(input_dim=INPUT_DIM, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
criterion_reg = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Model Configuration:")
print(f"  Input Dimension: {INPUT_DIM}")
print(f"  Hidden Dimension: 256")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Epochs: {EPOCHS}\n")

# ---- Training Loop ----
print("Training started...\n")
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
    
    avg_loss = epoch_loss / len(train_dl)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("\nTraining complete!\n")

# ---- Evaluation on Dev Set ----
print(f"{'='*60}")
print("EVALUATION ON DEVELOPMENT SET")
print(f"{'='*60}\n")

model.eval()
with torch.no_grad():
    dev_e, dev_p, dev_emp = model(X_dev.to(device))
    
    # Emotion MAE
    mae_emotion = mean_absolute_error(dev_df[COL_EMO], dev_e.cpu().squeeze())
    
    # Empathy MAE
    mae_empathy = mean_absolute_error(dev_df[COL_EMP], dev_emp.cpu().squeeze())
    
    # Polarity metrics
    preds_polarity = dev_p.argmax(dim=1).cpu().numpy()
    true_polarity = dev_df[COL_POL].values
    accuracy_polarity = accuracy_score(true_polarity, preds_polarity)
    
    print("📊 RESULTS:\n")
    print(f"Emotion Intensity (MAE):     {mae_emotion:.4f}")
    print(f"Empathy Intensity (MAE):     {mae_empathy:.4f}")
    print(f"Polarity Classification Accuracy: {accuracy_polarity:.4f}\n")
    
    print("Detailed Polarity Classification Report:")
    print(classification_report(true_polarity, preds_polarity, 
                                target_names=["Negative", "Neutral", "Positive"],
                                digits=4))

# ---- Predict on Test Set ----
print(f"\n{'='*60}")
print("GENERATING TEST PREDICTIONS")
print(f"{'='*60}\n")

with torch.no_grad():
    te_e, te_p, te_emp = model(X_test.to(device))
    te_e   = te_e.cpu().squeeze().numpy()
    te_emp = te_emp.cpu().squeeze().numpy()
    te_pol = te_p.argmax(dim=1).cpu().numpy()

# Use provided id column if present
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

out_csv = os.path.join(OUT_DIR, f"predictions_ann_{EMB_SUFFIX}.csv")
pred_df.to_csv(out_csv, index=False)
print(f"✅ Saved test predictions to: {out_csv}")
print(f"   Total predictions: {len(pred_df)}\n")

print(f"{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}\n")