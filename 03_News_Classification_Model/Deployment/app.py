import torch
import torch.nn as nn
import numpy as np
import pickle
import os 
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import FastText
import uvicorn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

class TextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_classes=4):
        super(TextClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attention_weights = torch.tanh(self.attention(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        x = self.fc1(context)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

print("\n🔄 Loading models...")

try:
    BASE_DIR = r"E:\ML - GITHUB\Text Classification\Models"
    checkpoint = torch.load(
    os.path.join(BASE_DIR, 'complete_checkpoint.pth'), 
    map_location=device
)
    embedding_matrix = checkpoint['embedding_matrix']
    
    model = TextClassifier(embedding_matrix, hidden_dim=128, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise

try:
    vocab_path = os.path.join(BASE_DIR, 'vocab.pkl')
    print(f"Loading vocab from: {vocab_path}")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("✅ Vocabulary loaded!")
except Exception as e:
    print(f"❌ Vocab load failed: {e}")
    raise

try:
    ft_path = os.path.join(BASE_DIR, 'fasttext_model.bin')
    print(f"Loading FastText from: {ft_path}")
    
    ft_model = FastText.load(ft_path)
    print("✅ FastText model loaded!")
except Exception as e:
    print(f"⚠️ FastText load warning: {e}")
    ft_model = None

class_mapping = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = " ".join(text.split())
    return text

def encode_text(text, vocab, max_len=200):
    if hasattr(vocab, 'encode'):
        encoded = vocab.encode(text)
    else:
        tokens = text.split()
        encoded = [vocab.get(token, 1) for token in tokens]
    
    if len(encoded) < max_len:
        encoded = encoded + [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    
    return encoded

def predict_news(text, model, vocab, device, max_len=200):
    cleaned_text = clean_text(text)
    encoded = encode_text(cleaned_text, vocab, max_len)
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = class_mapping[predicted.item()]
    confidence_score = float(confidence.item())
    
    return class_name, confidence_score

app = FastAPI(
    title="AG News Classifier API",
    description="News articles ko classify karne ke liye PyTorch LSTM + Attention",
    version="1.0"
)

class NewsInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    class_name: str
    confidence: float

@app.get("/")
def read_root():
    return {
        "message": "AG News Classifier API Live!",
        "usage": "POST /predict with {'text': 'your news text'}"
    }

@app.get("/health")
def health_check():
    return {
        "status": "Online",
        "model": "Loaded",
        "device": str(device),
        "classes": len(class_mapping)
    }

@app.get("/classes")
def get_classes():
    return {
        "available_classes": class_mapping,
        "total": len(class_mapping)
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(news: NewsInput):
    try:
        if not news.text or len(news.text.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Text minimal 5 characters hona chahiye"
            )
        
        class_name, confidence = predict_news(news.text, model, vocab, device)
        
        return PredictionOutput(
            class_name=class_name,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict-batch")
def predict_batch(news_list: list[NewsInput]):
    results = []
    
    for news in news_list:
        try:
            class_name, confidence = predict_news(news.text, model, vocab, device)
            results.append({
                "text": news.text[:50] + "...",
                "class_name": class_name,
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "text": news.text[:50] + "...",
                "error": str(e)
            })
    
    return {
        "total": len(results),
        "predictions": results
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting AG News Classifier API")
    print("="*60)
    print("Server: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )