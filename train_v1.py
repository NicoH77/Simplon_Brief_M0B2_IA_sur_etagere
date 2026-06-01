import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd


df = pd.read_csv("breast-cancer-train.csv") # Import des données du fichier CSV dans un dataframe pandas

# df.head(3) # Affichage des 3 premières lignes du fichier

# 
X = df.drop('target', axis=1) # création d'un dataframe contenant uniquement les features
y = df['target'] # Création d'un dataframe contenant uniquement la target

# print("Dataset Breast Cancer")
# print(f"Nombre d'échantillons : {X.shape[0]}") # Calcul du nombre de lignes (échantillons) du dataframe
# print(f"Nombre de features    : {X.shape[1]}")  # Calcul du nombre de colonnes (features) du dataframe


# ------------------------------------------------------------------------------
# Entrainement du modèle
# ------------------------------------------------------------------------------
# Utilisation de la fonction train_test_split de sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # taille de l'échantillon de test : 20%
    random_state=42,    # permet de garantir la reproductibilité du split
    stratify=y          # la classe étant déséquilibrée (357 1 pour 202 0), il garantit que la répartition des classes est respectée entre train et test
)

# print(f"Train : {X_train.shape[0]} échantillons") # Nombre d'échantillons dans le set d'entrainement
# print(f"Test  : {X_test.shape[0]} échantillons")  # Nombre d'échantillons dans le set de test

# ------------------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = X_train
# X_test_scaled = X_test

# ------------------------------------------------------------------------------
# Entrainement
# ------------------------------------------------------------------------------

model = LogisticRegression(max_iter=1000, random_state=42) # création du model
model.fit(X_train_scaled, y_train) # entrainement du modèle

y_pred   = model.predict(X_test_scaled) # utilise le modèle entraîné (fit) pour prédire la classe des données de test
accuracy = model.score(X_test_scaled, y_test) # Evalue la performance du modèle en comparant le résultat des prédictions et la classe de test

# print(f"\nAccuracy sur le test set : {accuracy:.4f}") # formattage accuracy avec 4 décimales
# print("\nRapport de classification :")
# print(classification_report(y_test, y_pred)) # rapport détaillé de performance du modèle


# print("Matrice de confusion :")
# print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------------------
# Sauvegarde
# ------------------------------------------------------------------------------

SCALER_PATH = "scaler.joblib"
MODEL_PATH  = "model.joblib"
 
# Le scaler et le modèle sont sauvegardés séparément.
# Les deux sont nécessaires pour faire une prédiction correcte.
joblib.dump(scaler, SCALER_PATH) # génère le fichier scaler.joblib
joblib.dump(model,  MODEL_PATH) # génère le fichier model.joblib


# Vérification : rechargement et inférence sur un exemple

# loaded_scaler = joblib.load(SCALER_PATH)
# loaded_model  = joblib.load(MODEL_PATH)
 
# sample_raw    = X_test[0:1]          # données brutes : récupère la première ligne du jeu de test
# print(sample_raw)
# sample_scaled = loaded_scaler.transform(sample_raw)  # normalisation de l'échantillon
# prediction    = loaded_model.predict(sample_scaled)  # prédiction
# proba         = loaded_model.predict_proba(sample_scaled) # indique le niveau de confiance du modèle

# print(proba)
# prediction, y_test[0:1]



