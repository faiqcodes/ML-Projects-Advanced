# 📧 Email Spam Detector API

An end-to-end Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)**. Built using Scikit-Learn, FastAPI, and Dockerized for production deployment.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.10
* **Machine Learning:** Scikit-Learn (Random Forest, TF-IDF Transformer)
* **Accuracy:** This is model is giving an accuracy of 97% with Random-Forest Algorithm 
* **API Framework:** FastAPI, Uvicorn
* **Containerization:** Docker
* **Serialization:** Joblib

---

## 📂 Project Structure

```text
02_Email_Spam_Detector/
│
├── Deployment/          # FastAPI application (app.py) & Dockerfile
├── models/              # Saved serialized models (.pkl files)
└── Notebook/            # Jupyter notebooks for EDA and model training