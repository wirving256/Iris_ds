import joblib

model = joblib.load("Gaussiannb_model.joblib")

def gnb(X):
    pred = model.predict(X)
    return pred

if __name__ == "__main__":
    X = [[4,1,2,2],[5,4,1,1]]
    result = gnb(X)
    print(result)