import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/train.csv")
X = df.drop("Survived", axis=1)

# Define column groups
num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked']

# Define pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Save transformed features
X_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)
X_df.to_csv("data/processed/train_preprocessed.csv", index=False)
print("✅ Preprocessing complete. CSV saved!")

# Save preprocessor for API use
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("✅ Preprocessor saved to models/preprocessor.pkl")