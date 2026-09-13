# 🏠 House Price Prediction

## Overview
It is a Advanced Machine Learning model for predicting house prices using LightGBM Algorithm. Deployed as a production-ready FastAPI application with Docker containerization.

**Model Performance:** 82.16% accuracy (R² Score: 0.8216)

---

## 📊 Project Highlights

| Metric | Value |
|--------|-------|
| **Algorithm** | LightGBM (Gradient Boosting) |
| **R² Score** | 0.8970 ( 89.70% ) |
| **RMSE** | 28105.33 |
| **Dataset Size** | 1,460 training samples |
| **Features** | 80+ engineered features |
| **Validation** | 5-Fold Cross Validation |

---

## 🎯 Best Model Hyperparameters

```python
n_estimators = 610
learning_rate = 0.03
max_depth = -1
num_leaves = 63
subsample = 0.7
colsample_bytree = 0.6
reg_alpha = 0.1
reg_lambda = 0.1
```

---

