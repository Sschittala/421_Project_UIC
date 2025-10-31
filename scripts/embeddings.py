# scripts/embeddings.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os, csv
import gensim.downloader as api

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

def generate_glove_embeddings(split: str, glove_model) -> None:
    """Generate GloVe embeddings by averaging word vectors."""
    df = read_split(split)
    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {split} CSV")

    texts = df["text"].astype(str).tolist()
    embeddings = []
    
    for text in texts:
        words = text.lower().split()
        word_vecs = [glove_model[word] for word in words if word in glove_model]
        
        if word_vecs:
            # Average word vectors to get sentence embedding
            sent_vec = np.mean(word_vecs, axis=0)
        else:
            # If no words found, use zero vector
            sent_vec = np.zeros(glove_model.vector_size)
        
        embeddings.append(sent_vec)
    
    embeddings = np.array(embeddings)
    output_path = os.path.join(OUT_DIR, f"{split}_embeddings_glove.npy")
    np.save(output_path, embeddings)
    print(f"Saved {split} GloVe embeddings: {embeddings.shape}")

def generate_sbert_embeddings(split: str) -> None:
    """Generate Sentence-BERT embeddings."""
    df = read_split(split)
    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {split} CSV")

    texts = df["text"].astype(str).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=True)

    output_path = os.path.join(OUT_DIR, f"{split}_embeddings_sbert.npy")
    np.save(output_path, emb)
    print(f"Saved {split} Sentence-BERT embeddings: {emb.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING GLOVE EMBEDDINGS")
    print("=" * 60)
    print("Loading GloVe model (glove-wiki-gigaword-100)...")
    glove_model = api.load("glove-wiki-gigaword-100")  # 100-dim vectors
    print(f"GloVe model loaded: {len(glove_model)} words, {glove_model.vector_size} dimensions\n")
    
    for split in ["train", "dev", "test"]:
        generate_glove_embeddings(split, glove_model)
    
    print("\n" + "=" * 60)
    print("GENERATING SENTENCE-BERT EMBEDDINGS")
    print("=" * 60)
    for split in ["train", "dev", "test"]:
        generate_sbert_embeddings(split)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - *_embeddings_glove.npy (100-dim)")
    print("  - *_embeddings_sbert.npy (384-dim)")
    print("\nYou can now train ANN models with both embedding types!")