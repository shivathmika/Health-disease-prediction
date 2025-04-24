from flask import Flask, render_template, request, redirect
import cv2
import os
from keras.models import load_model
from flask_ngrok import run_with_ngrok
import pickle
import numpy as np

import os
os.chdir("/content/drive/MyDrive/Health Disease Risk assessment/doctor_dashboard-main/templates")
 
app = Flask(__name__)
run_with_ngrok(app)

with open('/content/drive/MyDrive/Health Disease Risk assessment/Diabeties/Models/xgboost_BEST.pkl', 'rb') as file:
    diabetiesModel = pickle.load(file)
    
lungNodulesmodel = load_model("/content/drive/MyDrive/Health Disease Risk assessment/Lung nodule detection/Models/lung_nodule_CNN.h5")
breastCancerModel = load_model("/content/drive/MyDrive/Health Disease Risk assessment/breast cancer detection/Models/bestBreastprediction.h5")

@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template("signup")


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
    
    flash("Account created successfully. Please login.", "success")
    return redirect(url_for('login'))

@app.route('/login')
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
    flash("Login successful", "success")
    return redirect(url_for('dashboard'))
    
@app.route('/dashboard')
def dashboard():
    if request.method == "POST":
        return render_template('dashboard.html', username='vishnu')

@app.route('/diabetes')
def diabetes():

    return render_template('diabetes.html', username=session['username'])

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data_text = request.form.get('data')
    values = [float(i) for i in data_text.split(',')]
    prediction = diabetiesModel.predict(values)
    predictions.append({
            'input': data_text,
            'prediction': prediction
        })

    return render_template("diabetes_result.html", predictions=predictions)

@app.route('/lung-cancer')
def lung():
    return render_template('lung.html', username='vishnu')

@app.route('/predict/lung-cancer', methods=['GET', 'POST'])
def predict_lung_cancer_route():
    if request.method == 'POST':
        if 'images' not in request.files:
            return "No file part", 400

        file = request.files['file']
        image = file.read()
        # Run your model prediction here
        prediction = lungNodulesmodel.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction)

        results.append((file, predicted_class))

    return render_template('lung_result.html', results=results)

@app.route('/breast')
def breast():
    return render_template('breast.html', username='vishnu')

@app.route('/predict/breast', methods=['POST'])
def predict_breast():
    if request.method == 'POST':
        if 'images' not in request.files:
            return "No file part", 400

        file = request.files['file']
        image = file.read()
        
        prediction = breastCancerModel.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction)

        predictions.append({'prediction': prediction})

    return render_template('breast_result.html', predictions=predictions)
    

if __name__ == "__main__":
    app.run()
    
