
# Training and saving of the funnel neural network

#Library import
import pandas as pd
import numpy as np
import json
import joblib
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset loading
df = pd.read_csv("data/Final_Dataset_Mvp.csv", sep=";")

# Data preprocessing
target_col = "result"
df[target_col] = df[target_col].map({"winner": 1, "loser": 0})

cols_to_drop = [
    "goals", "goals conceded", "score",
    "goals conceded while last defender",
    "shots", "shooting percentage", "shots conceded"
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
X = df.select_dtypes(include=["number"]).copy()
y = X[target_col].copy()
X = X.drop(columns=[target_col])
feature_names = X.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Funnel Neural Network
model = Sequential([
    Dense(512, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    verbose=1
)

# Save everything
os.makedirs("models", exist_ok=True)

# The model
model.save("models/funnel_model.keras")

# The scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Name of the statistics
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("✅ Model, scaler and feature names saved correctly.")
