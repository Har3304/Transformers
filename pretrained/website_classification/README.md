# Website Classification using Transformer (TensorFlow)

## Overview

This project implements a Transformer-based neural network for website category classification using TensorFlow and Keras.

The model learns from cleaned website text and predicts one of several categories, including:

* Adult
* Business/Corporate
* Computers and Technology
* E-Commerce
* Education
* Food
* Forums
* Games
* Health and Fitness
* Law and Government
* News
* Photography
* Social Networking and Messaging
* Sports
* Streaming Services
* Travel

The architecture includes:

* Token Embedding
* Positional Encoding
* Multi-Head Self Attention
* Feed Forward Network
* Layer Normalization
* Dropout Regularization
* Global Average Pooling and Global Max Pooling
* Dense Classification Layers

---

## Dataset

The dataset should contain at least two columns:

* `cleaned_website_text`
* `Category`

Example:

| cleaned_website_text               | Category                 |
| ---------------------------------- | ------------------------ |
| football world cup news and scores | Sports                   |
| watch movies and tv shows online   | Streaming Services       |
| buy smartphones and laptops        | Computers and Technology |

---

## Features

* Transformer built from scratch
* Custom positional encoding layer
* Multi-head attention
* Label encoding with one-hot targets
* Stratified train-validation split
* Early stopping
* Dropout regularization
* L2 regularization
* Model serialization support
* Tokenizer saving and loading
* Label encoder saving and loading

---

## Model Architecture

Input Text

↓

Tokenizer

↓

Embedding + Positional Encoding

↓

Transformer Block

↓

Transformer Block

↓

Global Average Pooling

*

Global Max Pooling

↓

Concatenate

↓

Dense (64)

↓

Dense (32)

↓

Softmax Output Layer

---

## Training

Run:

```python
python train.py
```

The model is trained using:

* Adam optimizer
* Learning rate = 1e-4
* Batch size = 32
* Early stopping on validation loss

---

## Saved Files

After training:

```
website_classifier.keras
tokenizer.pkl
label_encoder.pkl
```

These files are required for inference.

---

## Inference Example

```python
text = """
Buy the latest smartphones, laptops and accessories.
Huge discounts available.
"""

predict_category(text)
```

Example output:

```
Predicted Category: Computers and Technology
Confidence: 96.42%
```

---

## Project Structure

```
.
├── website_classification.csv
├── train.py
├── inference.py
├── website_classifier.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## Libraries Used

* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Pickle

---

## Future Improvements

* DistilBERT fine-tuning
* RoBERTa
* Class weights
* ReduceLROnPlateau
* Data augmentation
* Hyperparameter optimization
* Attention visualization

---

## License

MIT License
