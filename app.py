import joblib,numpy as np
from flask import Flask,request,jsonify

model=joblib.load("model_dump.pkl")
app=Flask(__name__)

@app.route("/")
def home():
    return "Welcome to prediction"

@app.route("/POST",methods=["POST"])
def predict():
    data=request.get_json()
    try:
        if "features" not in data or len(data["features"])!=4:
            return jsonify({"Error": "Not enough parameters"}),400

        features=np.array(data["features"],dtype=float).reshape(1,-1)
        prediction=model.predict(features)[0]
        classes=["setosa","versicolor","virginica"]
        result={"Prediction": classes[prediction]}
        return jsonify(result)


    except Exception as e:
        return jsonify({"Error":e}),400
           