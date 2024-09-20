from flask import Flask,render_template,request
app = Flask(__name__)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('dat.csv')
X=data.drop('Crops',axis=1)
y=data['Crops']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=8)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        n = request.form.get("n")
        p = request.form.get("k")
        k = request.form.get("p")
        pH = request.form.get("pH")
        global model
        res = model.predict(np.array([[n, p, k, pH]]))
        print(res)
        return render_template("index.html", res=res[0])

    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)
