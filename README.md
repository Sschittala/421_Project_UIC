# CS 421 - Conversational Emotion & Empathy Prediction

**Natural Language Processing - Project Part 1 & Part 2**  
**University of Illinois Chicago**

Multi-task deep learning models for predicting and generating empathetic conversational responses.

---

## 📋 Project Overview

This project consists of two parts:

### Part 1: Emotion Classification Task
Build and compare models to predict emotion intensity, emotional polarity, and empathy scores from conversational text.

- **Q1:** Artificial Neural Networks (ANNs) with GloVe and Sentence-BERT embeddings
- **Q2:** Recurrent Neural Networks (RNN-LSTM) with learned embeddings
- **Q3:** Fine-tuned BERT transformer model
- **Q4:** Few-shot prompting with Large Language Models (Claude)
- **Q5:** Qualitative evaluation and model comparison

### Part 2: Conversational Generation Task
Implement two methods to generate next utterances in multi-turn conversations and compare their effectiveness.

- **Q1:** Corpus-Based Chatbot (retrieve similar utterances from training data)
- **Q2:** In-Context Learning with LLMs (few-shot prompting for generation)
- **Q3:** Qualitative evaluation and comparative analysis

### Dataset: TRAC-2 Conversational Empathy

- **Training:** 10,941 conversational turns
- **Development:** 990 turns
- **Test:** 2,316 turns
- **Tasks:** 
  - Emotion Intensity (1-5 regression)
  - Emotional Polarity (Negative/Neutral/Positive classification)
  - Empathy Intensity (1-5 regression)

---

## 🏆 Part 1: Results Summary

| Model | Emotion MAE | Empathy MAE | Polarity Acc | Training Time |
|-------|-------------|-------------|--------------|---------------|
| ANN-GloVe | 0.55 | 0.77 | 63.2% | 2 min (CPU) |
| ANN-SBERT | 0.53 | 0.75 | 65.1% | 2 min (CPU) |
| RNN-LSTM | 0.55 | 0.78 | 57.5% | 5 min (GPU) |
| **BERT** | **0.50** ✅ | **0.60** ✅ | **72.7%** ✅ | 15 min (GPU) |

**Winner:** BERT achieves best performance on all three tasks.

---

## 🏆 Part 2: Results Summary

### Q1: Corpus-Based Chatbot Performance

**Overall Rating: 3.95/5 - Good for Short Interactions, Limited for Sustained Dialogue**

| Metric | Score | Notes |
|--------|-------|-------|
| **Fluency** | 4.8/5 | Perfect (real corpus sentences) |
| **Relevance** | 3.4/5 | Excellent turns 6-7, poor turns 8-10 |
| **Coherence** | 3.1/5 | Excellent initially, breaks with repetition |
| **Emotion Alignment** | 4.5/5 | Good match for emotional context |

**Development Set Metrics:**
- ROUGE-1: 0.1281 (low but expected for retrieval)
- ROUGE-2: 0.0157
- ROUGE-L: 0.0996
- BLEU: 0.0000 (expected - too strict)
- BERTScore F1: 0.8484 ✅ (high semantic similarity)

**Why Low ROUGE but High BERTScore?**
- Corpus utterances use different vocabulary than gold references
- But they mean the same thing (semantic equivalence)
- This is CORRECT behavior for corpus retrieval

### Q2: In-Context Learning with LLMs

Status: Completed by teammate (see their submission)

### Q3: Qualitative Evaluation

**Corpus Method Ratings Across 25 Utterances:**

| Turn | Fluency | Relevance | Coherence | Emotion Align | Overall |
|------|---------|-----------|-----------|---------------|---------|
| Turn 6 | 5.0 | 4.8 | 4.8 | 4.2 | 4.7 |
| Turn 7 | 4.4 | 4.6 | 4.6 | 4.4 | 4.5 |
| Turn 8 | 4.8 | 3.4 | 2.6 | 4.6 | 3.9 |
| Turn 9 | 4.8 | 2.4 | 1.8 | 4.6 | 3.4 |
| Turn 10 | 4.8 | 1.8 | 1.6 | 4.6 | 3.2 |
| **Average** | **4.8** | **3.4** | **3.1** | **4.5** | **3.95** |

**Key Finding:** Quality degrades significantly after turn 7 due to utterance repetition when emotional context remains similar.

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
│   ├── ann_model.py                  # ANN architecture
│   └── bert_model_trained.pth        # Trained BERT weights
│
├── scripts/                          # Training scripts & notebooks
│   ├── embeddings.py                 # Generate embeddings
│   ├── train_ann.py                  # Train ANN models
│   ├── extract_conversations_q4.py   # Extract Q4 data
│   ├── Q3_Transformer.ipynb          # BERT training (Colab)
│   ├── RNN_Model.ipynb               # RNN training (Colab)
│   └── Q1_Corpus_Chatbot.ipynb       # Part 2 Q1 (Colab)
│
├── outputs/                          # Generated outputs
│   ├── PART 1 OUTPUTS:
│   │   ├── predictions_ann_glove.csv     # ANN-GloVe test predictions
│   │   ├── predictions_ann_sbert.csv    # ANN-SBERT test predictions
│   │   ├── Predictions_bert.csv         # BERT test predictions
│   │   ├── RNN_Predictions.csv          # RNN test predictions
│   │   ├── dev_predictions_*.csv        # Dev set evaluations
│   │   └── *_embeddings_*.npy           # Pre-computed embeddings
│   │
│   └── PART 2 OUTPUTS:
│       ├── generations_corpus.csv    # Q1 test predictions (335 rows)
│       ├── dev_generations.csv       # Q1 dev set generations
│       ├── dev_metrics.csv           # Q1 evaluation metrics
│       ├── generations_icl.csv       # Q2 test predictions (670 rows)
│       └── q3_corpus_analysis_CORRECTED.csv  # Q3 analysis data
│
├── Q4_conversations.txt              # Q4 selected conversations
├── LLM_output.txt                    # LLM predictions
└── README.md                         # This file
```

---

## PART 1: Classification Models

### 📊 Q1: Artificial Neural Networks

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

**Results:**
- ANN-GloVe: 63.2% polarity accuracy
- ANN-SBERT: 65.1% polarity accuracy

**Key Finding:** SBERT (384-dim) significantly outperforms GloVe (100-dim) - pre-trained embeddings better than static embeddings.

---

### 📊 Q2: RNN-LSTM

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

**Results:**
- RNN-LSTM: 57.5% polarity accuracy

**Key Finding:** Underperformed ANNs despite more complex architecture - learned embeddings worse than pre-trained with limited data (10K samples).

---

### 📊 Q3: BERT Transformer (Best Model)

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

**Results:**
- Emotion MAE: 0.50 ✅ (best)
- Empathy MAE: 0.68 ✅ (best)
- Polarity Accuracy: 72.7% ✅ (best)

**Key Finding:** Best performance across all tasks - pre-trained contextual embeddings are superior to learned or static embeddings.

---

### 📊 Q4: Few-Shot LLM (Claude)

**Approach:**
- Zero-shot prompting with task descriptions
- JSON output format
- Processes 5 conversations (25 turns)

**Results:** Competitive on empathy (MAE 0.73) but systematic bias on polarity due to annotation conventions.

---

### 📊 Q5: Qualitative Evaluation & Model Comparison

Detailed analysis of all models with failure cases and insights.

---

## PART 2: Conversational Generation

### Q1: Corpus-Based Chatbot

**Approach:** Retrieve most similar utterance from training corpus

**Algorithm:**
1. Load all 11,090 training utterances as corpus
2. For each test conversation, generate turns 6-10:
   - Build conversation history from previous turns
   - Calculate 4-component similarity score:
     - **Text Similarity (40%):** Cosine similarity of SentenceTransformer embeddings
     - **Emotion Similarity (20%):** 1 - |desired_emotion - corpus_emotion|
     - **Empathy Similarity (20%):** 1 - |desired_empathy - corpus_empathy|
     - **Polarity Match (20%):** 1 if polarities match, 0 otherwise
   - Retrieve utterance with highest weighted similarity
   - Add to conversation history for next turn

**Embedding Model:** SentenceTransformer (all-MiniLM-L6-v2)
- Dimension: 384
- Pre-trained on sentence similarity tasks

**Strengths:**
✅ Perfect grammar (real corpus sentences)
✅ Natural language (no AI artifacts)
✅ High semantic relevance (BERTScore 0.85+)
✅ Interpretable (can see source utterance)
✅ Fast inference (no GPU needed)

**Weaknesses:**
❌ Limited vocabulary (11,090 corpus utterances insufficient)
❌ Utterance repetition (same "best match" for similar emotions)
❌ No novel generation (constrained to corpus)
❌ Breaks conversation flow after 2-3 turns

**Output:** `generations_corpus.csv` (335 predictions: 67 conversations × 5 turns)

---

### Q2: In-Context Learning with LLMs

**Approach:** Few-shot prompting with pre-trained language model

**Prompt Structure:**
```
You are an empathetic conversation participant. Given conversation 
history and desired emotional context, generate the next response.

Example 1: [Conversation + Response]
Example 2: [Conversation + Response]
Example 3: [Conversation + Response]

Now, for this conversation:
[Conversation history]
Generate the next response:
```

**Configurations Tested:**
- 3-shot: 3 example conversations
- 5-shot: 5 example conversations

**Expected Strengths:**
- Novel utterances (not limited to corpus)
- High diversity (avoids repetition)
- Contextually appropriate
- Flexible emotional context adaptation

**Expected Weaknesses:**
- Potential grammatical errors
- Possible hallucinations
- Slower inference
- Less interpretable

**Output:** `generations_icl.csv` (670 predictions: 67 conversations × 10 turns)

---

### Q3: Qualitative Evaluation

**Objective:** Compare corpus and ICL methods on 5 selected conversations using human-informed evaluation.

**Selected Conversations:**
- Conv 68: Refugee/displacement crisis
- Conv 72: Mass shooting (analytical tone)
- Conv 74: Explosion (strong empathy)
- Conv 80: Orangutan endangerment
- Conv 85: Haiti disaster

**Evaluation Criteria (1-5 Likert Scale):**

1. **Fluency** - Grammatical correctness and natural flow
2. **Relevance** - How well response addresses conversation
3. **Coherence** - Consistency with previous utterances
4. **Emotion Alignment** - Match with expected emotional context

**Corpus Method Results:**

| Aspect | Corpus Based |
|--------|--------------|
| Fluency | 5 |
| Relevance | 3 |
| Coherence | 3 |
| Emotion Alignment | 4 |

**Key Findings:**

1. **Perfect Fluency (5/5):** Real corpus sentences guarantee grammar
2. **Good Initial Relevance (4.7/5 for turns 6-7):** Excellent opening responses
3. **Critical Weakness - Repetition:** Identical utterances retrieved for turns 8-10 when emotional context similar
4. **Coherence Breakdown (1.6/5 by turn 10):** Unnatural repetition destroys conversation
5. **Strong Emotion Alignment (4.5/5):** Polarity predictions ~71% accurate

**Root Cause of Repetition:**
- Corpus size: 11,090 utterances
- Required: 5 distinct responses per conversation × 67 conversations
- When turns 8-10 have similar emotional context (polarity/empathy), similarity algorithm finds same "best match"
- This is a fundamental limitation of corpus retrieval with limited vocabulary

**Verdict:** Corpus method excellent for short interactions (first 2 turns), unsuitable for extended dialogue.

---

## 🚀 Quick Start

### Part 1: Running Classification Models

#### Prerequisites
```bash
pip install torch transformers sentence-transformers
pip install pandas numpy scikit-learn tqdm
```

#### Generate Embeddings
```bash
python3 scripts/embeddings.py
```

#### Train ANN Models
```bash
# GloVe embeddings
python3 scripts/train_ann.py  # Edit: EMBEDDING_TYPE = "glove"

# Sentence-BERT embeddings
python3 scripts/train_ann.py  # Edit: EMBEDDING_TYPE = "sbert"
```

#### Train RNN (Google Colab)
1. Upload `RNN_Model.ipynb` to Colab
2. Run all cells
3. Download `RNN_Predictions.csv`

#### Train BERT (Google Colab)
1. Upload `Q3_Transformer.ipynb` to Colab
2. Enable GPU
3. Run all cells
4. Save trained weights: `torch.save(model.state_dict(), 'bert_model_trained.pth')`

---

### Part 2: Running Generation Models

#### Q1: Corpus-Based Chatbot (Google Colab)
1. Upload `Q1_Corpus_Chatbot.ipynb` to Colab
2. Upload dataset files
3. Run all cells
4. Download `generations_corpus.csv`

#### Q2: In-Context Learning (Google Colab)
1. Upload `Project_Part_2_Question_2.ipynb` to Colab
2. Run all cells
3. Download `generations_icl.csv`

#### Q3: Qualitative Evaluation
- Analyze both outputs using evaluation rubric
- Rate on 1-5 Likert scale
- Generate comparative report

---

## 🔍 Key Insights

### Part 1: Classification

1. **Pre-training Matters More Than Architecture**
   - ANN-SBERT (65%) > RNN-LSTM (57%)
   - Pre-trained embeddings beat learned embeddings

2. **BERT Dominates**
   - Contextual embeddings adapt word meanings
   - Attention focuses on emotion-bearing words
   - Pre-training on 3.3B words provides strong baseline

3. **Task Difficulty Hierarchy**
   - Emotion Intensity: MAE 0.50-0.62 (easiest)
   - Emotional Polarity: 57-73% accuracy (moderate)
   - Empathy Intensity: MAE 0.60-0.84 (hardest)

4. **Common Failure Mode**
   - Models struggle with empathic statements about tragedies
   - Focus on explicit negative words instead of pragmatic meaning

5. **Class Imbalance Issues**
   - Negative polarity only 17% of data
   - Low recall on negative class (~30%)
   - Underestimate extreme values (≥4.0)

### Part 2: Generation

1. **Vocabulary Constraint is Real**
   - Corpus method perfect for first 2 turns
   - By turn 8-10, vocabulary exhaustion apparent

2. **Fluency Without Relevance is Insufficient**
   - Perfect grammar doesn't make good dialogue
   - Relevance and coherence equally important

3. **Emotion Alignment Achievable**
   - Polarity predictions 71% accurate
   - Corpus method respects emotional context

4. **Trade-off Identification**
   - Corpus: Safety (perfect grammar) vs Flexibility (limited diversity)
   - LLM: Flexibility (novel) vs Safety (potential errors)
   - Hybrid approach optimal

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
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore metrics

---

## 📈 Evaluation Metrics

### Part 1 Metrics

**Emotion & Empathy (Regression):**
- Metric: Mean Absolute Error (MAE)
- Range: 1.0 to 5.0
- Goal: Minimize MAE

**Emotional Polarity (Classification):**
- Metric: Accuracy & F1-score
- Classes: Negative (0), Neutral (1), Positive (2)
- Goal: Maximize accuracy

### Part 2 Metrics

**Automatic Metrics:**
- ROUGE-1/2/L: Lexical overlap
- BLEU: N-gram precision
- BERTScore: Semantic similarity

**Human Evaluation:**
- Fluency (1-5)
- Relevance (1-5)
- Coherence (1-5)
- Emotion Alignment (1-5)

---

## 🎯 Future Improvements

### Part 1
1. Class balancing for negative examples
2. More epochs (10-15 for BERT instead of 5)
3. Ensemble methods combining all models
4. Conversation-level modeling using context

### Part 2
1. Hybrid approach: Corpus (turns 1-2) + LLM (turns 3-5)
2. Context-aware retrieval to avoid repetition
3. Corpus expansion to 50K+ utterances
4. Fine-tune embeddings on emotion data
5. Multi-rater evaluation for reliability

---

## 🛠️ Troubleshooting

### GPU/Memory Issues
- Reduce batch size (16 → 8)
- Use CPU for ANN training

### Import Errors
```bash
pip install sentence-transformers
pip install transformers torch
```

### CSV Parsing Errors
- Dataset uses escaped quotes
- Use `escapechar='\\'` in `pd.read_csv()`

### Reproducibility
```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## 📝 File Formats

### Prediction Files (CSV)

**Format:**
```csv
id,Emotion,EmotionalPolarity,Empathy
1,2.5,1,3.2
2,1.8,0,1.5
```

- **id:** Unique identifier
- **Emotion:** Float [1.0-5.0]
- **EmotionalPolarity:** Integer [0/1/2]
- **Empathy:** Float [1.0-5.0]

### Generation Files (CSV)

**Part 2 Q1 Format:**
```csv
id,turn_number,generated_response
67,6,"I think we should..."
67,7,"You made a good point..."
```

**Part 2 Q2 Format:**
```csv
id,turn_number,generated_response
67,6,"I understand your concern..."
67,7,"That's an important perspective..."
```

---

## 📄 License

Academic project for CS 421 - Natural Language Processing at UIC.

---

**Authors:** Nishanth Nagendran & Teammate  
**Course:** CS 421 - Natural Language Processing  
**University:** University of Illinois Chicago  
**Date:** December 2025

---

## References

- **SentenceTransformer:** Sentence-BERT paper (Reimers & Gupta, 2019)
- **TRAC-2 Dataset:** Conversational Empathy Workshop
- **BERT:** Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- **BERTScore:** Zhang et al. (2020)
- **GloVe:** Pennington et al. (2014)