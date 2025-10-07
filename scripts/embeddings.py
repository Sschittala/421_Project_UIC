from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

DATA_DIR = "../P1_DATA"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_embeddings(split):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(f"{DATA_DIR}/trac2_CONVT_{split}.csv")
    texts = df["text"].astype(str).tolist()
    emb = model.encode(texts, show_progress_bar=True)
    np.save(f"{OUT_DIR}/{split}_embeddings.npy", emb)
    print(f"Saved {split} embeddings:", emb.shape)

for split in ["train", "dev", "test"]:
    generate_embeddings(split)
