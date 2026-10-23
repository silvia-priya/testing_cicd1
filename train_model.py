from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data=load_iris()
X,y=data.data,data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rfc=RandomForestClassifier(n_estimators=10,random_state=42)

rfc.fit(X_train,y_train)
joblib.dump(rfc,"model_dump.pkl")
print("Model loaded")

