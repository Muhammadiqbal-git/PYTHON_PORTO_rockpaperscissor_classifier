import os
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import models, layers
from keras.applications.mobilenet_v2 import preprocess_input
class Inference():
    def inference():
        img_path = os.path.join(os.getcwd(), "inference_imgs", "rock.jpg")

        # img = keras.preprocessing.image.load_img(img_path)
        # img = keras.preprocessing.image.img_to_array(img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        plt.imshow(img)
        plt.show()

        img = tf.convert_to_tensor(img)
        img = preprocess_input(img)

        # img = img/255
        print(img.shape)

        img = np.expand_dims(img, axis=0)
        print(img.shape)
        labels = ["paper", "rock", "scissor"]

        model = models.load_model("model/out_model.model")
        predict = model.predict(img)
        print(labels[np.argmax(predict)])

    def predict(img_path) -> str:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = tf.convert_to_tensor(img)
        img = np.expand_dims(img, axis=0)
        # img = img/255
        img = preprocess_input(img)

        labels = ["paper", "rock", "scissor"]
        model = models.load_model("model/out_model.model")
        predict = model.predict(img)
        return labels[np.argmax(predict)]


