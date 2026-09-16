from fastapi import FastAPI 
import joblib
from pydantic import BaseModel
app = FastAPI(title="Email Spam Detector app .")

model = joblib.load("Random_Forest_Model.pkl")
tfidf = joblib.load("tfidf.pkl")
features_names = joblib.load("Feature_names.pkl")

class EmailText(BaseModel):
    email_content : str

@app.get("/")
def home():
    return {"message": "API is Working !"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: EmailText):
    text = data.email_content.lower().split()
    feature_vector = [text.count(word) for word in features_names]

    tfidf_data = tfidf.transform([feature_vector])
    prediction = model.predict(tfidf_data)

    return {"prediction" : "This Email is Spam" if prediction[0] == 1 else"This Email is'nt Spam"}

