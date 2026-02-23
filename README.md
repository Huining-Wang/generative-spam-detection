# LLM Spam Detection: Bayesian Inverse Classification

This repository contains the implementation and analysis of a spam detection system using Large Language Models (LLMs). It leverages a **decoder-only transformer** and applies **Bayesian Inverse Classification** to determine whether an email is spam or ham based on sequence likelihoods.

## Project Highlights
* **Methodology:** Implemented zero-shot learning, naive prompting, and full fine-tuning using an autoregressive Llama-based model.
* **Top Performance:** Achieved an accuracy of **0.98455** on the Kaggle public leaderboard through rigorous hyperparameter tuning.
* **Core Insight:** Demonstrated that Bits Per Character (BPD) is a more reliable metric for model quality than accuracy, especially after accuracy saturates.

## Model & Approach

### Decoder-Only Transformer & KV Cache
The model generates text autoregressively. At each step, it predicts the next token based on previous tokens:
$$P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})$$

To optimize inference, **KV caching** is implemented. It stores previously computed key/value pairs, reducing the decoding cost from quadratic to linear. While it significantly speeds up generation, it requires memory that grows linearly with sequence length.

### Bayesian Inverse Classification
Instead of outputting a simple "right" or "wrong" label, the model compares the log-likelihood of two constructed prompts for each email:
1. `[Template] + "Ham" + [Email Content]`
2. `[Template] + "Spam" + [Email Content]`

The log-likelihood is computed as:
$$\log P(x | y) = \sum_{t=1}^{T} \log P(x_t | x_{<t}, y)$$

A softmax function is applied over the two log-likelihood values, and the label with the higher likelihood is chosen as the final prediction.

## Experiments & Fine-Tuning

### Hyperparameter Tuning
Extensive experiments were conducted to understand the impact of learning rate, iterations, and batch size. All parameters of the pretrained model were updated to minimize the negative log-likelihood per character.

* **Learning Rate:** A moderate learning rate of $2 \times 10^{-5}$ provided the best trade-off between training stability and progress. (Higher rates led to instability; lower rates caused slow convergence).
* **Iterations:** The model achieved its optimal performance at **420 iterations**. Training beyond 450 iterations led to overfitting, where BPD started to increase and Kaggle accuracy dropped.

### Relationship Between BPD and Accuracy
During training, accuracy rapidly saturated at 1.0. However, the model continued to improve as **BPD (Bits Per Character) continued to decrease**. The checkpoint with the lowest BPD directly corresponded to the highest performance on the Kaggle leaderboard, proving BPD to be the primary indicator of true model quality.

## How to Run

The codebase provides several examples demonstrating different inference strategies. Ensure you have the `uv` package manager installed.

**1. General Text Generation**
```bash
uv run -m examples.chatbot_example
```
**2. Zero-Shot Baseline**
Evaluates the model without extra training:
```bash
uv run -m examples.bayes_inverse --method zero_shot
```
**3. Naive Prompting**
Injects a richer prompt at inference time:
```bash
uv run -m examples.bayes_inverse --method naive_prompting
```
**4. Full Fine-Tuning**
Triggers the full parameter update before evaluation (this was used to achieve the final Kaggle score):
```bash
uv run -m examples.bayes_inverse --method full_finetune
