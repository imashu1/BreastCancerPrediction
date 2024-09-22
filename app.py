import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import json
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    new_data = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(new_data)
    # Convert output to a dictionary
    output_dict = {'prediction': output.item()} 
    print(output_dict)
    # Convert numpy scalar to Python scalar
    return jsonify(output_dict)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=model.predict(final_input)[0]
    if(output==1):
        return render_template("popup.html",prediction_text="Congratulation! The cell type is Benign",prediction_text_second="You are safe")
    else:
        return render_template("popup.html",prediction_text="Sorry! The cell type is Malignant",prediction_text_second="You need medicines")
if __name__=="__main__":
    app.run(debug=True)