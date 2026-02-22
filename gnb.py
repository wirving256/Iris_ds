from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import joblib

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Number of mislabeled points {(y_test != y_pred).sum()} out of a total points : {X_test.shape[0]}")
joblib.dump(model, "Gaussiannb_model.joblib")
