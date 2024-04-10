from flask import Flask, render_template, request
import cv2
import tensorflow as tf
from inference import Inference
from matplotlib import pyplot

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def template():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["img"]
    image_path = "./imgs/predict.jpg"
    image.save(image_path)
    out = Inference.predict(image_path)
    return render_template('index.html', output=out, fname = image.filename)




if __name__ == "__main__":
    app.run(port=3000, debug=True)