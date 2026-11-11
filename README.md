# deep-learning-project
# Dynamic Sequence-to-Sequence (Seq2Seq) with TensorFlow

An implementation of a basic dynamic Sequence-to-Sequence (Seq2Seq) architecture using TensorFlow. This project demonstrates how an encoder-decoder model processes variable-length sequences with padding, masking, and dynamic recurrent neural networks (RNNs).

---

## 📌 Project Overview

The objective of this implementation is a **memorization & sequence reproduction task**:
* The model receives a random sequence of integers of variable length[cite: 4].
* The encoder processes the time-major padded sequence using an LSTM cell and passes the final state to the decoder[cite: 4].
* The decoder uses the encoder's final state and generates the target sequence token-by-token (with `<EOS>` end-of-sequence delimiter)[cite: 4].

---

## 🏗️ Architecture & Pipeline

1. **Batch Generation & Padding (`gen_batch`)**[cite: 4]:
   * Converts variable-length input sequences into time-major padded arrays (`PAD = 0`)[cite: 4].
   * Tracks individual sequence lengths for dynamic computation[cite: 4].

2. **Embedding Layer**[cite: 4]:
   * Maps sparse integer tokens into dense continuous vectors using `tf.nn.embedding_lookup`[cite: 4].

3. **Encoder**[cite: 4]:
   * `tf.contrib.rnn.LSTMCell` with 20 hidden units[cite: 4].
   * `tf.nn.dynamic_rnn` processes time-major inputs to extract `encoder_final_state`[cite: 4].

4. **Decoder**[cite: 4]:
   * Initialized with `encoder_final_state`[cite: 4].
   * Computes predictions via linear projection over vocabulary size[cite: 4].

5. **Optimization**[cite: 4]:
   * Loss: Stepwise Softmax Cross-Entropy with Logits[cite: 4].
   * Optimizer: Adam Optimizer[cite: 4].

---

## 📊 Model Parameters

| Parameter | Value |
| :--- | :--- |
| **Vocabulary Size** | 10[cite: 4] |
| **Embedding Size** | 20[cite: 4] |
| **Encoder Units** | 20 (LSTM)[cite: 4] |
| **Decoder Units** | 20 (LSTM)[cite: 4] |
| **Batch Size** | 100[cite: 4] |
| **Training Steps** | 3000 batches[cite: 4] |

---

## 📈 Training Results

The model converges quickly, dropping from an initial cross-entropy loss of ~`2.29` down to ~`0.09` after 3,000 steps[cite: 4]:

* **Step 0**: Minibatch Loss: `2.29` (Random predictions)[cite: 4]
* **Step 1000**: Minibatch Loss: `0.30`[cite: 4]
* **Step 2000**: Minibatch Loss: `0.13`[cite: 4]
* **Step 3000**: Minibatch Loss: `0.088` (Accurately reproduces sequences with `<EOS>`)[cite: 4]

---

## 🛠️ Requirements & Setup

This implementation relies on TensorFlow 1.x APIs (`tf.contrib` and static graph sessions)[cite: 4]:

```bash
pip install tensorflow==1.15.0 numpy matplotlib seaborn
