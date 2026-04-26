# NLP-Text-Classification-Model-Evaluation

This project focuses on classifying English text into six basic emotions using a Multi-Layer Perceptron (MLP). 

## 📌 Project Overview
The goal of this project is to build a machine learning model capable of detecting emotions in short text snippets. The project involves data preprocessing, feature extraction using TF-IDF, hyperparameter optimization, and evaluating the model's robustness against out-of-domain data.

- **Corpus:** `dair-ai/emotion` (Twitter messages).
- **Labels:** Sadness (0), Joy (1), Love (2), Anger (3), Fear (4), Surprise (5).
- **Model Architecture:** Multi-Layer Perceptron (MLP).
- **Feature Extraction:** TF-IDF Vectorization.

## 📊 Dataset Information
The dataset is based on the paper **"CARER: Contextualized Affect Representations for Emotion Recognition"** (Saravia et al., 2018). It contains Twitter data labeled via "Distant Supervision" (using hashtags).

| Subset | Size |
| :--- | :--- |
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

## 🚀 Methodology
1. **Preprocessing:** Text cleaning and tokenization.
2. **Vectorization:** `TfidfVectorizer` with a limit of 5,000 features and English stop-word removal.
3. **Modeling:** Scikit-learn's `MLPClassifier`.
4. **Optimization:** Explored different hidden layer architectures and learning rates. The best model used a high learning rate (0.01) and 200 iterations for convergence.

## 📈 Results
The final optimized MLP model achieved the following performance on the test set:

- **Weighted F1-Score:** 0.85
- **Accuracy:** 0.85

### Performance Summary
The model performs exceptionally well on high-frequency classes like **Joy** and **Sadness** (F1-scores ~0.88-0.89) but shows room for improvement in minority classes like **Surprise** and **Love**.

### Relation to State-of-the-Art (SOTA)
While SOTA models (Transformers like RoBERTa or DeBERTa) achieve **93%-95%** F1-scores, this MLP approach provides a highly efficient and lightweight alternative, training in seconds with respectable performance.

## 🧪 Bonus: Out-of-Domain Evaluation
To test the model's generalization, I created a custom dataset of **50 news headlines** (formal, objective language).
- **Result:** F1-score dropped to **0.18**.
- **Insight:** This significant drop illustrates **Domain Shift**. A model trained on casual social media slang struggles to interpret the formal and indirect emotional expressions found in journalism.

## 🛠️ Requirements
To run the notebook, you need the following libraries:
- `datasets`
- `scikit-learn`
- `pandas`
- `matplotlib`

You can install them via pip:
```bash
pip install datasets scikit-learn pandas matplotlib
