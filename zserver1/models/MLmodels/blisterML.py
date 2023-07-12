import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pylab as plt
import tensorflow_hub as hub
import os
# print(os.getcwd())
import numpy as np


def current_dir():
    print(os.getcwd())
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    return os.getcwd()

model = tf.keras.models.load_model("models/MLmodels/blister")
# halfModel = tf.keras.models.load_model('F:\Year 4\CDAP\Model training\saved_model\my_halfmodel')

def prepare_image(img):
    # img = Image.open(io.BytesIO(img))
    img = img
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = img/255
    return img


def predict_result(img):
    img = img/255
    predicted = model.predict(img)
    predicted = np.argmax(predicted, axis=1)
    print(predicted)
    if predicted == 0:
        return 'Acetazolamide'
    elif predicted == 1:
        return 'Candesartan'
    elif predicted == 2 or predicted == 3 or predicted == 4 or predicted == 5 or predicted == 66:
        return 'Amlodipine'
    elif predicted == 9 or predicted == 10 or predicted == 33 or predicted == 34 or predicted == 35 or predicted == 36:
        return 'Aspirin'
    elif predicted == 11 or predicted == 12 or predicted == 46 or predicted == 53:
        return 'Atorvastatin'
    elif predicted == 24 or predicted == 15 or predicted == 16 or predicted == 17 or predicted == 18 or predicted == 75:
        return 'Bisoprolol'
    elif predicted == 7:
        return 'Captopril'
    elif predicted == 21 or predicted == 22 or predicted == 23 or predicted == 29 or predicted == 51:
        return 'Clopidogrel'
    elif predicted == 30 or predicted == 31 or predicted == 32 or predicted == 76:
        return 'Diltiazem'
    elif predicted == 37 or predicted == 38 or predicted == 39:
        return 'Enalapril'
    elif predicted == 40 or predicted == 44:
        return 'Ezetimibe'
    elif predicted == 41:
        return 'Irbesartan'
    elif predicted == 6:
        return 'Isosorbide Mononitrate'
    elif predicted == 8 or predicted == 25 or predicted == 43 or predicted == 50 or predicted == 74:
        return 'Losartan'
    elif predicted == 13 or predicted == 14 or predicted == 52 or predicted == 55 or predicted == 56:
        return 'Metoprolol'
    elif predicted == 47 or predicted == 48 or predicted == 49:
        return 'Nifedipine'
    elif predicted == 19 or predicted == 54:
        return 'Ramipril'
    elif predicted == 27 or predicted == 28 or predicted == 42 or predicted == 57 or predicted == 58 or predicted == 59 or predicted == 60 or predicted == 61 or predicted == 62 or predicted == 63 or predicted == 64 or predicted == 72 or predicted == 73:
        return 'Rosuvastatin'
    elif predicted == 65:
        return 'Simvastatin'
    elif predicted == 20 or predicted == 26 or predicted == 45 or predicted == 67 or predicted == 68 or predicted == 69 or predicted == 70 or predicted == 71:
        return 'Telmisartan'
    else:
        return 'This medicine cannot be identified'


# def predict_halfresult(img):
#     predicted = halfModel.predict(img)
#     predicted = np.argmax(predicted, axis=1)
#     print(predicted)
#     if predicted == 0:
#         return 'Acetazolamide'
#     elif predicted == 1:
#         return 'Amlodipine'
#     elif predicted == 2 or predicted == 3 or predicted == 11 or predicted == 12:
#         return 'Aspirin'
#     elif predicted == 4 or predicted == 14 or predicted == 16:
#         return 'Atorvastatin'
#     elif predicted == 5 or predicted == 6 or predicted == 8 or predicted == 9:
#         return 'Bisoprolol'
#     elif predicted == 7:
#         return 'Clopidogrel'
#     elif predicted == 13:
#         return 'Irbesartan'
#     elif predicted == 15:
#         return 'Nifedipine'
#     elif predicted == 10 or predicted == 17 or predicted == 18 or predicted == 19:
#         return 'Rosuvastatin'
#     else:
#         'This medicine cannot be identified'