import os
import threading
import numpy as np
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, redirect, url_for, request, flash
from pyngrok import ngrok

app = Flask(__name__)
port = "5000"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

with open('/content/drive/MyDrive/Health Disease Risk assessment/Diabeties/Models/random_forest_BEST.pkl', 'rb') as file:
    model = pickle.load(file)
    
lungNodulemodel = tf.keras.models.load_model('/content/drive/MyDrive/Health Disease Risk assessment/LND - experimentation/Models/bestModel.h5')

breastCancerModel = tf.keras.models.load_model('/content/drive/MyDrive/Health Disease Risk assessment/breast cancer detection/Models/bestBreastprediction.h5')

# ... Update inbound traffic via APIs to use the public-facing ngrok URL
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if(username == 'vvardh15' and password == 'Optum@123'):
            return redirect(url_for('dashboard'))
        
        else:
            return render_template('login.html', fav = "Credentials are incorrect, please try again")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', username="Shivatmika")
    
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html', username='Shivatmika')
    
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data_text = request.form.get('data')
    
    k = []
    k = data_text.split(',')
    l = []
    for i in k:
        l.append(float(i))
    data = np.array(l)
    
    data = data.reshape(1, -1)
    
    k = int(model.predict(np.array(data))[0])
    predictions = []
    if k == 0:
        predictions.append({
            'input': 'non-diabetic',
            'prediction': 'Low'
        })
    else:
        predictions.append({
            'input': 'Diabetic',
            'prediction': 'High'
        })
    
    return render_template('diabetes_result.html', predictions = predictions)

@app.route('/lung-cancer')
def lung():
    return render_template('lung.html', username='Shivatmika')

@app.route('/predict/lung-cancer', methods=['GET', 'POST'])
def predict_lung_cancer_route():
    if request.method == 'POST':
        files = request.files.getlist('images')
        
        if not files:
            return "No files uploaded", 400
        
        for file in files:
            filename = file.filename
        
        file = files[0]  # ✅ get the first file

        img = Image.open(file).convert("L")  

        img_arr = np.array(img).astype(np.float32)  
        img_arr = img_arr[..., np.newaxis]

        original_min = -1024
        original_max = 100

        img_arr = img_arr / 255.0
        img_arr = img_arr * (original_max - original_min) + original_min
        
        print(img_arr)

        prediction = lungNodulemodel.predict(np.expand_dims(img_arr, axis=0))
        predicted_class = np.argmax(prediction)
        result = 'Patient belongs to the benign segment'
        if predicted_class == 0:
            result = 'Patient belongs to the benign segment'
        else:
            result = 'Patient belongs to the Malignent segment, CANCER DETECTED'
        
        results = []
        
        results.append({
            filename: filename,
            result : result
        }) 
    return render_template('lung_result.html', results = results)

@app.route('/breast-cancer')
def breast():
    return render_template('breast.html', username= 'Shivatmika')

@app.route('/predict/breast', methods=['POST'])
def predict_breast():
    if request.method == 'POST':
        files = request.files.getlist('images')
        
        for file in files:
            filename = file.filename
        
        img = Image.open(file).convert("L")  

        img_arr = np.array(img).astype(np.float32)  
        img_arr = img_arr[..., np.newaxis]
        
        predictions = breastCancerModel.predict(np.expand_dims(img_arr, axis=0))
        predicted_class = np.argmax(predictions)
        
        if predicted_class == 0:
            predict = 'malignant'
        elif predicted_class == 1:
            predict = 'benign'
        else:
            predict = 'normal'
        
        predictions = []
        predictions.append({
            "filename": filename,
            "prediction" : predict
        })
        
        return render_template('breast_result.html', predictions = predictions)

@app.route('/logout')
def logout():
    return render_template('login.html')
    
@app.route('/about')
def about():
    return render_template('Aboutus.html')
        
# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

