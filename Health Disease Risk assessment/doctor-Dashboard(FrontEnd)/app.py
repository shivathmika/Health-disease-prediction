from flask import Flask, render_template, request, redirect
import cv2
import os
from keras.models import load_model
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

app.config['IMAGE_UPLOADS'] = "/content/drive/MyDrive/LND"
model = load_model("/content/drive/MyDrive/LND/facefeatures_new_model.h5")

@app.route('/', methods = ['GET', 'POST'])
def hello():
    return render_template("upload.html")


@app.route('/upload-image', methods = ['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files['image']
            image_filename = "Lungs.png"
            image.save(os.path.join(app.config['IMAGE_UPLOADS'],image_filename))
    return redirect('/predict')

@app.route('/predict')
def predict():
    image = cv2.imread("/content/drive/MyDrive/LND/Lungs.png")
    IMG_SIZ = 224
    img = cv2.resize(image,(IMG_SIZ,IMG_SIZ),3)
    img.shape = (1,224,224,3)
    if(model.predict(img)[0][0]==1):
      p = "affected"
    else:
      p = "not affected"
    return render_template("upload.html", prediction_text = "the status of this patient is: {}".format(p))

if __name__ == "__main__":
    app.run()