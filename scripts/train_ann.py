import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, classification_report
import numpy as np
import pandas as pd
from models.ann_model import ANNModel

train_data = pd.read_csv("../P1_DATA/trac2_CONVT_train.csv")
dev_data = pd.read_csv("../P1_DATA/trac2_CONVT_dev.csv")

X_train = torch.tensor(np.load("../outputs/train_embeddings.npy"), dtype=torch.float32)
X_dev = torch.tensor(np.load("../outputs/dev_embeddings.npy"), dtype=torch.float32)

y_train_emotion = torch.tensor(train_data["emotion"].values, dtype=torch.float32).unsqueeze(1)
y_train_empathy = torch.tensor(train_data["empathy"].values, dtype=torch.float32).unsqueeze(1)
y_train_polarity = torch.tensor(train_data["polarity"].values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train_emotion, y_train_polarity, y_train_empathy)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

model = ANNModel()
criterion_reg = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, y_e, y_p, y_emp in train_dl:
        optimizer.zero_grad()
        pred_e, pred_p, pred_emp = model(xb)
        loss = criterion_reg(pred_e, y_e) + criterion_cls(pred_p, y_p) + criterion_reg(pred_emp, y_emp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_dl):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    pred_dev_e, pred_dev_p, pred_dev_emp = model(X_dev)
    mae_e = mean_absolute_error(dev_data["emotion"], pred_dev_e.squeeze())
    mae_emp = mean_absolute_error(dev_data["empathy"], pred_dev_emp.squeeze())
    preds_p = pred_dev_p.argmax(dim=1)
    print("Emotion MAE:", mae_e)
    print("Empathy MAE:", mae_emp)
    print("Polarity report:\n", classification_report(dev_data["polarity"], preds_p))
