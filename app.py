from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

iris = load_iris()

# ✅ Use ONLY sepal features (first 2 columns)
X = iris.data[:, :2]
y = iris.target

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

class_names = ["setosa", "versicolor", "virginica"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    # ✅ Only use sepal features for prediction
    features = np.array([[sl, sw]])
    pred = int(model.predict(features)[0])
    return {"prediction": pred, "class_name": class_names[pred]}
