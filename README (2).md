# titanic_survivor-

---

# Titanic Survival Prediction Web App

## Project Overview

This project predicts whether a passenger aboard the Titanic would survive, based on historical passenger data. It uses a trained classification model deployed via Flask, allowing users to input passenger details through a web interface and get real-time survival predictions.

---

## Problem Statement

Using features like age, sex, ticket class, and embarkation point, the goal is to predict the survival status (`Survived: 0 or 1`) of a passenger. This binary classification task is based on the famous Titanic dataset provided by Kaggle.

---

## Tech Stack

- **Language**: Python  
- **Libraries**: pandas, numpy, scikit-learn, joblib  
- **Model Used**: Logistic Regression (or RandomForest, if updated)  
- **Web Framework**: Flask  
- **Frontend**: HTML, CSS (Bootstrap optional)

---

## Directory Structure

```
TitanicPrediction/
│
├── Data/                  # Original and processed CSVs
│   ├── train.csv
│   └── cleaned.csv
│
├── Training/              # Jupyter notebooks
│   └── preprocessing.ipynb
│   └── model_training.ipynb
│
├── App/
│   ├── app.py
│   ├── model/
│   │   └── titanic_model.pkl
│   ├── templates/
│   │   └── index.html
│   └── static/ (if styled)
│
├── requirements.txt
├── README.md
```

---

## Model Training Summary

- Encoded categorical features (Sex, Embarked)
- Imputed missing values (Age, Embarked)
- Scaled numerical values (Age, Fare)
- Target column: `Survived`
- Final model saved as `titanic_model.pkl`

---

## How to Run the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI server
```bash
cd App
python app.py
```

### 3. Access locally in browser
```
http://127.0.0.1:5000/docs
```

---

## Inputs Expected

Form fields:
- Passenger Class (1/2/3)
- Sex (encoded as 0 for female, 1 for male)
- Age
- SibSp (siblings/spouses aboard)
- Parch (parents/children aboard)
- Fare
- Embarked (encoded: C=0, Q=1, S=2)

Example:
```python
[3, 1, 22, 1, 0, 7.25, 2]
```

Output:
```
Predicted Outcome: Survived (1)
```

---

## Features

- Real-time survival prediction via browser
- Clean Bootstrap form interface
- Logistic Regression or RandomForest backend
- Modular codebase with separate training and deployment flows

---

## Future Enhancements

- Add dropdowns for categorical inputs (Sex, Embarked)
- Use SHAP or feature importance for interpretability
- Dockerize app for easy sharing and deployment
- Deploy on Heroku or Streamlit for public access

---


