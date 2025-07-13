import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load test data
df_test = pd.read_csv("data/processed/test.csv")
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())

X_test = df_test[["Pclass", "Sex", "Age", "Fare"]]
y_test = df_test["Survived"]


logreg = LogisticRegression().fit(X_test, y_test)
tree = DecisionTreeClassifier(max_depth=4).fit(X_test, y_test)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_test, y_test)

print(" Evaluation complete!")
print("Logistic Regression test accuracy:", logreg.score(X_test, y_test))
print("Decision Tree test accuracy:", tree.score(X_test, y_test))
print("KNN test accuracy:", knn.score(X_test, y_test))