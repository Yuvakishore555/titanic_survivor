from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load your data
X = pd.read_csv("data/processed/train_preprocessed.csv")
y = pd.read_csv("data/processed/train.csv")["Survived"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# ✅ Save the model to a file
joblib.dump(model, "models/logistic_model.pkl")