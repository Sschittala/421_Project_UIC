# CS 421 - Conversational Emotion & Empathy Prediction

**Natural Language Processing - Project 1**  
**University of Illinois Chicago**

Multi-task deep learning models for predicting emotion intensity, emotional polarity, and empathy in conversational text.

---

## 📋 Project Overview

This project implements and compares four modeling approaches for understanding emotions in conversations:
- **Q1:** Artificial Neural Networks (ANNs) with GloVe and Sentence-BERT embeddings
- **Q2:** Recurrent Neural Networks (RNN-LSTM) with learned embeddings
- **Q3:** Fine-tuned BERT transformer model
- **Q4:** Few-shot prompting with Large Language Models (Claude)
- **Q5:** Qualitative evaluation and model comparison

### Dataset: TRAC-2 Conversational Empathy

- **Training:** 10,941 conversational turns
- **Development:** 990 turns
- **Test:** 2,316 turns
- **Tasks:** 
  - Emotion Intensity (1-5 regression)
  - Emotional Polarity (Negative/Neutral/Positive classification)
  - Empathy Intensity (1-5 regression)

---

## 🏆 Results Summary

| Model | Emotion MAE | Empathy MAE | Polarity Acc | Training Time |
|-------|-------------|-------------|--------------|---------------|
| ANN-GloVe | 0.55 | 0.77 | 63.2% | 2 min (CPU) |
| ANN-SBERT | 0.53 | 0.75 | 65.1% | 2 min (CPU) |
| RNN-LSTM | 0.55 | 0.78 | 57.5% | 5 min (GPU) |
| **BERT** | **0.50** ✅ | **0.60** ✅ | **72.7%** ✅ | 15 min (GPU) |

**Winner:** BERT achieves best performance on all three tasks.

---

## 📁 Project Structure

```
421_Project_UIC/
├── P1_DATA/                          # Dataset files
│   ├── trac2_CONVT_train.csv
│   ├── trac2_CONVT_dev.csv
│   └── trac2_CONVT_test.csv
│
├── models/                           # Model architectures
│   ├── __init__.py
│   └── ann_model.py                  # ANN architecture
│
├── scripts/                          # Training scripts
│   ├── embeddings.py                 # Generate embeddings
│   ├── train_ann.py                  # Train ANN models
│   ├── extract_conversations_q4.py   # Extract Q4 data
│   └── Q3_Transformer.ipynb          # BERT training (Colab)
│   └── RNN_Model.ipynb               # RNN training (Colab)
│
├── outputs/                          # Generated outputs
│   ├── predictions_ann_glove.csv     # ANN-GloVe test predictions
│   ├── predictions_ann_sbert.csv     # ANN-SBERT test predictions
│   ├── Predictions_bert.csv          # BERT test predictions
│   ├── RNN_Predictions.csv           # RNN test predictions
│   ├── dev_predictions_*.csv         # Dev set predictions for Q5
│   ├── *_embeddings_*.npy            # Pre-computed embeddings
│   └── Q5_Report_Concise.md          # Final Q5 report
│
├── Q4_conversations.txt              # Q4 selected conversations
├── LLM_output.txt                    # LLM predictions
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install pandas numpy scikit-learn tqdm
```

### Step 1: Generate Embeddings

```bash
python3 scripts/embeddings.py
```

This creates:
- `train_embeddings_glove.npy` (10941 × 100)
- `train_embeddings_sbert.npy` (10941 × 384)
- Similar files for dev and test sets

### Step 2: Train ANN Models

**GloVe embeddings:**
```bash
# Edit train_ann.py: EMBEDDING_TYPE = "glove"
python3 scripts/train_ann.py
```

**Sentence-BERT embeddings:**
```bash
# Edit train_ann.py: EMBEDDING_TYPE = "sbert"
python3 scripts/train_ann.py
```

Outputs:
- `predictions_ann_glove.csv` (test predictions)
- `predictions_ann_sbert.csv` (test predictions)

### Step 3: Train RNN (in Google Colab)

1. Upload `RNN_Model.ipynb` to Colab
2. Upload dataset files to Colab
3. Run all cells
4. Download `RNN_Predictions.csv`

### Step 4: Train BERT (in Google Colab)

1. Upload `Q3_Transformer.ipynb` to Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Run all cells
4. Download `Predictions_bert.csv`

### Step 5: Extract Q4 Conversations

```bash
python3 scripts/extract_conversations_q4.py > Q4_conversations.txt
```

### Step 6: LLM Predictions (Optional)

Use Claude/ChatGPT with prompts in assignment to get predictions, save to `LLM_output.txt`.

---

## 📊 Model Details

### Q1: Artificial Neural Networks

**Architecture:**
- Input: Pre-computed embeddings (100-dim GloVe or 384-dim SBERT)
- Hidden layer: 256 neurons with ReLU + Dropout (0.3)
- Output heads:
  - Emotion: 1 neuron (MSE loss)
  - Polarity: 3 neurons (CrossEntropy loss)
  - Empathy: 1 neuron (MSE loss)

**Training:**
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Epochs: 50
- Device: CPU
- Time: ~2 minutes

**Key Finding:** SBERT (384-dim) significantly outperforms GloVe (100-dim) - 65% vs 63% polarity accuracy.

---

### Q2: RNN-LSTM

**Architecture:**
- Embedding layer: 128-dim (learned from scratch)
- LSTM: 256 hidden units, 2 layers, dropout 0.3
- Output heads: Same as ANN

**Training:**
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Epochs: 10
- Device: GPU (T4 in Colab)
- Time: ~5 minutes

**Key Finding:** Surprisingly underperformed ANNs (57% vs 65%) - learned embeddings worse than pre-trained with limited data.

---

### Q3: BERT Transformer

**Architecture:**
- Base model: `bert-base-uncased` (110M parameters)
- Max sequence length: 128 tokens
- Output heads added on top of [CLS] token
- Multi-task learning with combined loss

**Training:**
- Optimizer: AdamW (lr=1e-5)
- Batch size: 16
- Epochs: 5
- Scheduler: Linear warmup
- Device: GPU (T4 in Colab)
- Time: ~15 minutes

**Key Finding:** Best performance across all tasks - pre-trained contextual embeddings are superior.

---

### Q4: Few-Shot LLM (Claude)

**Approach:**
- Zero-shot prompting with task descriptions
- JSON output format
- Processes 5 conversations (25 turns)

**Key Finding:** Competitive on empathy (MAE 0.73) but systematic bias on polarity due to annotation conventions.

---

## 🔍 Key Insights

### 1. Pre-training Matters More Than Architecture

- ANN-SBERT (65%) > RNN-LSTM (57%) despite simpler architecture
- Pre-trained embeddings beat learned embeddings with limited data (10K samples)

### 2. BERT Dominates

- Contextual embeddings adapt word meanings based on context
- Attention mechanism focuses on emotion-bearing words
- Pre-training on 3.3B words provides strong baseline

### 3. Task Difficulty Hierarchy

- **Easiest:** Emotion Intensity (MAE 0.50-0.62)
- **Moderate:** Emotional Polarity (57-73% accuracy)
- **Hardest:** Empathy Intensity (MAE 0.60-0.84)

Empathy requires understanding social context beyond explicit words.

### 4. Common Failure Mode

All models struggle when empathic statements discuss tragedies. Example:
- Text: "Such a tragedy. My first reaction was sadness for them."
- True label: Positive (empathic validation)
- Models often predict: Negative (focus on "tragedy", "sadness")

### 5. Class Imbalance Issues

- Negative polarity: Only 17% of data
- All models have low recall on negative class (~30%)
- Models underestimate extreme emotion/empathy (≥4.0)

---

## 📈 Evaluation Metrics

### Emotion & Empathy (Regression)
- **Metric:** Mean Absolute Error (MAE)
- **Range:** 1.0 to 5.0
- **Goal:** Minimize MAE

### Emotional Polarity (Classification)
- **Metric:** Accuracy & F1-score
- **Classes:** Negative (0), Neutral (1), Positive (2)
- **Goal:** Maximize accuracy & F1

---

## 🎯 Future Improvements

### Short-Term
1. **Class balancing:** Oversample negative examples, use class weights
2. **More epochs:** BERT with 10-15 epochs instead of 5
3. **Ensemble:** Combine BERT + LLM predictions

### Long-Term
1. **Conversation-level modeling:** Use previous turns as context
2. **Dialogue-specific pre-training:** Fine-tune on Reddit/Twitter conversations
3. **Multi-annotator labels:** Reduce annotation noise
4. **Active learning:** Identify and label model's weak cases

---

## 📝 File Formats

### Prediction Files (CSV)

All prediction files have the same format:

```csv
id,Emotion,EmotionalPolarity,Empathy
1,2.5,1,3.2
2,1.8,0,1.5
...
```

- **id:** Unique turn identifier
- **Emotion:** Float [1.0-5.0]
- **EmotionalPolarity:** Integer [0=Negative, 1=Neutral, 2=Positive]
- **Empathy:** Float [1.0-5.0]

### Embedding Files (NumPy)

Saved as `.npy` files:
- GloVe: Shape (N, 100)
- SBERT: Shape (N, 384)

Load with: `np.load('train_embeddings_glove.npy')`

---

## 🛠️ Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in Colab notebooks (16 → 8)

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:** `pip install sentence-transformers`

### Issue: CSV parsing errors
**Solution:** Dataset uses escaped quotes. The `read_csv()` functions in scripts handle this with `escapechar='\\'`

### Issue: Different results on re-run
**Solution:** Set random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## 📚 Dependencies

### Core Libraries
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - BERT and tokenizers
- `sentence-transformers>=2.2.0` - SBERT embeddings
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical computing
- `scikit-learn>=1.2.0` - Metrics and preprocessing

### Optional
- `tqdm` - Progress bars
- `matplotlib` - Visualization

---

## 📄 License

Academic project for CS 421 - Natural Language Processing at UIC.

---

- **Dataset:** TRAC-2 Workshop on Conversational Empathy
- **Pre-trained Models:** 
  - GloVe embeddings (Stanford NLP)
  - Sentence-BERT (UKP Lab)
  - BERT (Google Research)
- **Course:** CS 421 taught at University of Illinois Chicago

---