from array import array
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pylab as plt
import tensorflow_hub as hub
import os
# print(os.getcwd())
import numpy as np
from tensorflow import keras

# model = tf.keras.models.load_model("d:/projects/medical_app/server/zserver1/models/MLmodels/pill_model")
model = tf.keras.models.load_model("models/MLmodels/pill_model")

def predict(image_path):
    labels =['Alphapress_1mg', 'Amoxycilin_250mg', 'Mefanamic_acid_500mg']
    prediction = model.predict(img)
    score = tf.nn.softmax(prediction[0])
    item = labels[np.argmax(score)]
    print(prediction)
    print(score)
    print(item)
    return item

def prepare_image(img):
    # img = Image.open(io.BytesIO(img))
    img = img
    img = img.resize((180, 180))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = img/255
    return img

if __name__ == "__main__":
    img = Image.open("d:/projects/medical_app/server/zserver1/models/MLmodels/1 Adiflam.jpeg")
    img = prepare_image(img)
    print(predict(img))
