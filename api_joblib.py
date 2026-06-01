import joblib
import numpy as np
from fastapi import FastAPI

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
app = FastAPI()

SCALER_PATH = "scaler.joblib"
MODEL_PATH  = "model.joblib"

# Chargement du modèle
loaded_scaler = joblib.load(SCALER_PATH)
loaded_model  = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok"}
 
@app.post("/predict")
def predict(data: dict):
    # Séparation features / target
    # Le CSV contient 31 colonnes : 30 features + 1 colonne "target"
    # On isole les features en supprimant la target — elle n'est pas utilisée pour la prédiction
    features = data["features"]   # 30 valeurs brutes
    target   = data["target"]     # vraie étiquette — ignorée ici, utile pour évaluation côté client
    
    
    # Compléter pour avoir "prediction" et "proba"
    # sample_raw    = features          # données brutes : récupère la première ligne du jeu de test
    sample_raw    = np.array(features).reshape(1, -1)            # reshape obligatoire
    sample_scaled = loaded_scaler.transform(sample_raw)          # normalisation de l'échantillon
    prediction    = loaded_model.predict(sample_scaled)[0]       # prédiction
    proba         = loaded_model.predict_proba(sample_scaled)[0] # indique le niveau de confiance du modèle
    
    return {
        "prediction":            int(prediction),
        "label":                 "malignant" if prediction == 0 else "benign",
        "probability_malignant": round(float(proba[0]), 4),
        "probability_benign":    round(float(proba[1]), 4),
    }