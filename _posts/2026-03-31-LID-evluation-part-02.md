---
layout: post
title: "Language identification models evaluation - part 02 - replicate the Wordllama detect"
date: 2026-03-31
category: data-science 
---

*Part 02 of the language identification models evaluation - replicate the WorldLamma detect model*


**Project period:** Sep 2025 - Mar 2026

**Project team:**
                        
- Mr.Hoan Nguyen (Superviser)
- Huynh Cao Khoi

**Role:**                       
- Data Collecting
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Evaluation
- Evaluation result analysis

**Tools:**

- Hugging Face API, Polars, sentence_splitter, pyspark,unicodedata, semhash, cleanlab, Matplotlib, Seaborn, pytorch

# **Overview**
In the previous part, we consider that the [llamadetect](https://github.com/dleemiller/WordLlamaDetect) model offers a well-balanced trade-off across accuracy, memory usage, and throughput. In this section, we will deeply dive into its structure.

**WordLlama Detect** is a WordLlama-based library specifically designed for language identification. 
It supports **148 languages** and is built for high accuracy with fast, CPU-only inference 
using NumPy — no GPU required.

Key highlights:
- **Model size:** only 13MB
- **Throughput:** 70,000 – 100,000 texts/second on a single thread
- **Language coverage:** 148 languages


For more details, we can visit the [office blog post](https://huggingface.co/blog/dleemiller/wordllama-detect)

The overall structure of WorldLLamma is look like this:

```plaintext
Input Text
    │
    ▼
┌────────────────┐
│   Tokenize     │
└────────────────┘
    │
    ▼
Token IDs: [t₁, t₂, ..., tₙ]
    │
    ▼
┌──────────────────────────┐
│  Lookup Embeddings       │ ← Gemma 3 frozen embeddings
└──────────────────────────┘
    │
    ▼
Embeddings: [e₁, e₂, ..., eₙ]  (each eᵢ ∈ ℝᵈ)
    │
    ▼
┌──────────────────────────┐
│  Project + Weight        │ ← Learned: W, b, {wᵢ}
└──────────────────────────┘
    │  ℓᵢ = wᵢ · (W·eᵢ + b)
    ▼
Logits: [ℓ₁, ℓ₂, ..., ℓₙ]  (each ℓᵢ ∈ ℝᴸ)
    │
    ▼
┌──────────────────────────┐
│  Log-Sum-Exp Pool        │
└──────────────────────────┘
    │  z = log[Σᵢ exp(ℓᵢ)]
    ▼
Aggregated Logits: z ∈ ℝᴸ
    │
    ▼
┌──────────────────────────┐
│     Softmax              │
└──────────────────────────┘
    │
    ▼
Language Probabilities
```

The most important component of LlamaDetect is the **embedding lookup step** — a rich, 
high-quality text embedding provides a strong input representation, directly impacting 
the model's identification accuracy.

This naturally raises an interesting question:

> *If we replace Gemma 3 with a stronger embedding model, can we boost LlamaDetect's performance?*

The goal of this experiment is to find an alternative embedding backbone that is either 
**faster** or **more accurate** than the current Gemma 3-based WordLlama.

We explored two main approaches:

- **Surveying multilingual LLMs and embedding models** — collecting a set of candidate 
  models that support multilingual text representation as potential replacements for Gemma 3.
- **Exploring alternative training strategies** — investigating whether changes to the 
  training process itself could improve performance independently of the backbone.




# **1. Surveying multilingual LLMs and embedding models**

For the language identification task, the quality of a model is closely tied to the 
diversity of its training data (the data mixture). The wider the range of languages and domains a model 
covers, the more expressive and discriminative its embeddings will be. Therefore, 
**language and domain coverage** was our primary criterion when surveying candidate models.


The full list of model is [here](https://docs.google.com/document/d/1qFTDiBIJfTtIjRBfQxz3F6okbqdQOtpfieh_T5ha3g8/edit?tab=t.pbv9n589vj3d)


## **Retraining the LlamaDetect Model**

Firstly, I firstly check the availability of the code provided in [WordLlama Detect's github](https://github.com/dleemiller/WordLlamaDetect), while running the provided code, I encountered serveral challenges:

- **Missing data preparation code**: The repository does not include the data preparation pipeline, so I had to reimplement it from scratch.

- **CUDA compatibility issues**: With the limitation of device, I used Kaggle GPU to train to model. However, the default configuration requires a modern CUDA device, which is incompatible with Kaggle's GPU environment. I had to set up a 
  custom environment and modify the package requirements accordingly.

The training process is shown in [this notebook](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)

During training, I discovered several key insights about the model:

- **High VRAM requirement**: the training process requires more than 16GB of VRAM, 
  which exceeds Kaggle's GPU capacity. I had to adjust the training hyperparameters 
  to fit within the available resources.
- **Sensitivity to hyperparameters** — even small changes in hyperparameters led to 
  noticeable differences in model performance, making tuning particularly challenging.

Ultimately, while I successfully trained the model, it was unable to match the 
performance of the author's released model. This is primarily due to hardware 
limitations — we were unable to replicate the full training configuration reported 
in the original work.


## **Training Worldllama Detect with other LLMs or embedding model**
After make sure we can use the code, I replace the Gemma3 model with some models in the survey:

|Model|Num support language|Num trained lang|Accuracy|F1-score|
|Baseline - [Gemma 3 27b](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|148|148|0.916073|0.917110|
|[Mala 500](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|| 169| 0.8438| 0.8423|
|[mmbert](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)| 1834|190|0.732565|0.737130|
|[Apertus 70b](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|1812|190|0.678821|0.676668|
|[Qwen 3](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|122|111|0.840024|0.839785|
|[Gemma 300m](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|148|148|0.913774|0.913658|
|[bge m3](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|31|25|0.997031|0.997030|
|[babel](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|25|22|0.994620|0.994624|
|[eurobert](https://www.kaggle.com/code/caokhoihuynh/run-wordllama-training?scriptVersionId=295608664)|15|13|0.989970|0.989981|

As the language support range is different between models, I also find the overall metric of the common languages across models:

| Model | Accuracy | F1-macro |
|---|---|---|
| gemma3_27b | 0.9282 | 0.9293 |
| mala_500 | 0.8870 | 0.8835 |
| mmbert | 0.8519 | 0.8634 |
| Apertus 70b | 0.8259 | 0.8272 |
| Qwen 3 | 0.9127 | 0.9025 |
| Gemma 300m | 0.9338 | 0.9364 |
| bge m3 | 0.9975 | 0.9972 |
| babel | — | — |
| eurobert | — | — |


| Comparison | Common Languages | Model | Accuracy | F1 Score |
|---|---|---|---|---|
| gemma3_27b vs apertus-70b | 144 | gemma3_27b | 0.9175 | 0.9161 |
| | | apertus-70b | 0.7247 | 0.7278 |
| gemma3_27b vs gemma_300m | 146 | gemma3_27b | 0.9182 | 0.9168 |
| | | gemma_300m | 0.9160 | 0.9152 |
| gemma3_27b vs mala_500 | 132 | gemma3_27b | 0.9145 | 0.9133 |
| | | mala_500 | 0.8956 | 0.8823 |
| gemma3_27b vs mmbert | 144 | gemma3_27b | 0.9174 | 0.9161 |
| | | mmbert | 0.8038 | 0.7971 |
| gemma3_27b vs qwen3 | 96 | gemma3_27b | 0.9254 | 0.9206 |
| | | qwen3 | 0.8803 | 0.8667 |
| gemma3_27b vs bge_m3 | 25 | gemma3_27b | 0.9293 | 0.9303 |
| | | bge_m3 | 0.9970 | 0.9970 |

For more detail, we can visit [this notebook](https://www.kaggle.com/code/caokhoihuynh/evaluation-result-wordllama)


1. bge_m3 is the clear winner Despite being evaluated on only 25 common languages, bge_m3 dramatically outperforms gemma3_27b with 0.997 accuracy vs 0.929. This suggests it has extremely strong embeddings for the languages it supports.

2. gemma_300m punches above its weight gemma_300m is nearly on par with gemma3_27b (0.916 vs 0.918 accuracy) despite being a much smaller model. This is a significant finding — you get similar performance at a fraction of the model size and computational cost.

3. qwen3 underperforms relative to its size qwen3 is evaluated on only 96 common languages and still trails gemma3_27b by ~4.5%. Given its size, this is a disappointing result for language identification.

4. apertus-70b and mmbert lag significantly Both models fall well behind gemma3_27b:

apertus-70b: -19% accuracy gap
mmbert: -11% accuracy gap
These models are likely not optimized for low-level token-level language discrimination.

