from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from  sklearn.preprocessing import StandardScaler
import pickle 

application = Flask(__name__)
app=application


ridge_model=pickle.load(open('/Users/kanwal/Desktop/Code/MLPROJECT/model/rig.pkl','rb'))
scaler=pickle.load(open('/Users/kanwal/Desktop/Code/MLPROJECT/model/scaler.pkl','rb'))





@app.route("/")
def index():
 return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
 if request.method=='POST':
  Temperature=float(request.form.get("Temperature"))
  RH=float(request.form.get("RH"))
  Ws=float(request.form.get("Ws"))
  Rain=float(request.form.get("Rain"))
  FFMC=float(request.form.get("FFMC"))
  DMC=float(request.form.get("DMC"))
  ISI=float(request.form.get("ISI"))
  Classes=float(request.form.get("Classes"))
  Region=float(request.form.get("Region"))
  new_data=[[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]
  new_data_scaled=scaler.transform(new_data)

  result1=ridge_model.predict(new_data_scaled)[0]
  return render_template('home.html',result=result1)
 else:
  return render_template("home.html")








if __name__=="__main__":

 app.run(port=5001)``