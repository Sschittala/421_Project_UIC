# scripts/embeddings.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os, csv

# resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../P1_DATA")
OUT_DIR  = os.path.join(BASE_DIR, "../outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def read_split(split: str) -> pd.DataFrame:
    """
    Tolerant CSV reader. Skips malformed lines with bad quotes or stray commas.
    Must be kept in sync with train_ann.py to ensure row alignment.
    """
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

def generate_embeddings(split: str) -> None:
    df = read_split(split)
    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {split} CSV. Got columns: {list(df.columns)}")

    texts = df["text"].astype(str).tolist()

    # sentence-level embeddings model (384-dim)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=True)

    # save embeddings; train_ann.py will read the CSV with the same parser,
    # so row indices will align
    np.save(os.path.join(OUT_DIR, f"{split}_embeddings.npy"), emb)
    print(f"Saved {split} embeddings:", emb.shape)

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        generate_embeddings(split)
