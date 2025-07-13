import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("data/processed/train.csv")
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

features = ["Pclass", "Sex", "Age", "Fare"]
X = df[features]
y = df["Survived"]

logreg = LogisticRegression()
logreg.fit(X, y)


tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

print("All models trained successfully!")
print("Logistic Regression accuracy:", logreg.score(X, y))
print("Decision Tree accuracy:", tree.score(X, y))
print("KNN accuracy:", knn.score(X, y))