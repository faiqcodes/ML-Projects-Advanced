# News Classification Model

## Overview
A deep learning model that classifies news articles into different categories using NLP techniques.

## Dataset
- **Source:** AG News Classification Dataset
- **Size:** Training Dataset(120000, 4) , Test Dataset(7600, 4)
- **Features:** News text data

## Model Architecture
- **Input:** News article text
- **Preprocessing:**  vocabulary building , Embedding 
- **Architecture:** FastText + LSTM Model with Attention
- **Output:** News category classification

## Results
- **Accuracy:** 91.12%
- **Precision:** ~92% (World), ~98% (Sports), ~90% (Business), ~91% (Sci/Tech)
- **Recall:**  ~90.6% (World), ~98.3% (Sports), ~89.8% (Business), ~90.3% (Sci/Tech)
- **F1-Score:** 91.10%

## Files
- `Text_Classification.ipynb` - Main notebook with model training
- `Datasets/` - Training and test data
- `Models/` - Saved model weights
- `Deployment/` - Production-ready code

## Dependencies
```bash
pip install torch pandas numpy scikit-learn
```

## Usage
```python
from model import NewsClassifier
classifier = NewsClassifier()
prediction = classifier.predict("Your news text here")
```

## Author
Muhammad Faiq (faiqcodes)
- GitHub: [@faiqcodes](https://github.com/faiqcodes)
