#  Positional Encoding in Transformers: A Comparative Study

##  Overview

This project implements a **Transformer encoder-decoder architecture from scratch in PyTorch** with a key focus on **modular positional encoding (PE)**. The goal is to evaluate how different positional encoding strategies affect model performance, generalization, and attention behavior.

We implement a **swappable positional encoding module**, allowing seamless comparison between:

* Sinusoidal Positional Encoding
* Learned Positional Encoding
* Rotary Positional Encoding (RoPE)

---

##  Objectives

1. Build a Transformer architecture **entirely from scratch**
2. Design a **plug-and-play positional encoding interface**
3. Pretrain models on **WikiText-2**
4. Fine-tune and evaluate on **AG News classification**
5. Compare:

   * Validation loss
   * Classification accuracy
   * Length generalization behavior

---

##  Architecture

The core model is a custom implementation:

###  Components

* Token Embedding Layer
* Multi-Head Self Attention (custom implementation)
* Feedforward Network (GELU activation)
* Layer Normalization
* Dropout Regularization
* Swappable Positional Encoding Module

###  Key Design Choice

```text
Embedding → Positional Encoding → Transformer Layers → Output Head
```

The positional encoding module is injected as:

```python
x, rotary_freqs = pe_module(x)
```

This enables:

* Standard additive PE (Sinusoidal / Learned)
* Frequency-based rotation (RoPE)

---

##  Positional Encoding Variants

### 1. Sinusoidal PE

* Deterministic, non-learned
* Generalizes well to longer sequences
* Injected additively into embeddings

### 2. Learned PE

* Trainable positional embeddings
* Often achieves lower training loss
* Limited extrapolation beyond training length

### 3. RoPE (Rotary Positional Encoding)

* Encodes position via **rotation in embedding space**
* Applied inside attention mechanism
* Strong balance between performance and generalization

---

##  Dataset

###  Pretraining

* **WikiText-2**
* Language modeling objective

###  Downstream Task

* **AG News Classification**
* 4 classes:

  * World
  * Sports
  * Business
  * Sci/Tech

---

##  Training Pipeline

### Phase 1 — Pretraining

* Objective: Next-token prediction
* Dataset: WikiText-2
* Model learns language structure + positional awareness

### Phase 2 — Fine-tuning

* Encoder reused as feature extractor
* Classification head added:

  ```python
  pooled = x.mean(dim=1)
  logits = Linear(pooled)
  ```
* Dataset: AG News

---

##  Implementation Challenges

### 1. Vocabulary Mismatch

Pretraining and downstream tasks used different vocabularies.

**Solution:**

* Input tokens were remapped using:

  ```python
  src = src % vocab_size
  ```
* Ensured compatibility with pretrained embeddings

---

### 2. CUDA Device-Side Errors

Frequent crashes due to invalid indices in embedding layer.

**Resolution:**

* Clamped / remapped indices
* Verified label ranges (0–3)

---

### 3. Modular PE Integration

Ensuring compatibility across:

* Additive PE (sinusoidal, learned)
* Multiplicative/rotational PE (RoPE)

---

##  Evaluation Metrics

* Cross-Entropy Loss
* Classification Accuracy
* Stability across sequence lengths

---

##  Key Observations

| Positional Encoding | Performance | Generalization | Notes                |
| ------------------- | ----------- | -------------- | -------------------- |
| Sinusoidal          | Moderate    | Excellent      | Strong extrapolation |
| Learned             | Best train  | Poor           | Overfits to length   |
| RoPE                | Strong      | Strong         | Best balance overall |

---

##  Length Generalization

Models were trained on shorter sequences and evaluated on longer ones:

* Sinusoidal PE maintained performance
* Learned PE degraded significantly
* RoPE showed robust behavior

---

##  Future Work

* Use shared vocabulary between pretraining and downstream tasks
* Add CLS-token based pooling
* Evaluate on larger datasets (e.g., GLUE benchmark)
* Integrate FlashAttention for efficiency

---

##  Conclusion

This experiment demonstrates that:

> **Positional encoding is not just a detail — it fundamentally shapes how Transformers understand sequence structure.**

Among the variants tested:

* **RoPE offers the best balance** between accuracy and generalization
* **Sinusoidal PE excels in extrapolation**
* **Learned PE performs well but lacks robustness**

---

##  Tech Stack

* PyTorch
* Kaggle Notebooks (GPU)
* Python 3.12

---

## 👤 Author

Dhruv Prakash, Kushal Bhattad, Moksha Chaitanya
Transformer Architecture Exploration Project

---
