import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pylab as plt
import tensorflow_hub as hub

# print(os.getcwd())

from tensorflow import keras
import json

from . import prescription_update

def current_dir():
    return os.getcwd()

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

model = tf.keras.models.load_model("models/MLmodels/prediction_model_v1")

charecters = ['y','(','\"','!','&','f',
                    ':','2','S','P','M','m','\'','J',')','T','j','h',',','9',
                    '/','C','l','7','Q',';','B','z','v','q','n','V','F','D','-','Z',
                    'b','X','a','r','p','?','K','4','u','E','O','e','o','5','x','R','.',
                    'H','8','Y','U','#','6','3','s','t','N','1','L',
                    '0','w','i','g','A','*','k','W','c','d','I','G','+'
                ]

char_to_num = keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=charecters, num_oov_indices=0, mask_token=None
    )
num_to_char = keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def distortion_free_resize(image, img_size):
    w, h = img_size
    
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def setup_model():
    model = tf.keras.models.load_model("models/MLmodels/prescription_model")

    prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    return prediction_model

def prepare_image(img):
    img = img
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # img = img/255
    # img = img.resize((64, 128))
    
    img = np.array(img)
    img = np.expand_dims(img, 0)
    
    return img

def predict_result(image):
    image_width = 128
    image_height = 64
    img_size = (image_width,image_height)
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    image = tf.cast(image, tf.float32) / 255.0
    img = image.numpy()
    img = np.expand_dims(img, 0)

    # prediction_model = setup_model()
    preds = model.predict(img)
    max_length = 21
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0][
            :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res.replace('[UNK]',''))

    return output_text

def check_word(word):
    with open("models/MLmodels/words.json", "r") as f:
        words = json.load(f)
    if word in words:
        return words[word]
    else:                   
        return None


def predict_result_update(image):
    image = cv2.resize(image,(800,600))
    boxes = prescription_update.predict(image)
    text = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(800,600))
    (_, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    gray = np.expand_dims(gray, 2)
    print(boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        try:
            y1, x1, y2, x2 = box[0]-10, box[1]-10, box[2]+10, box[3]+10
            img = gray[y1:y2, x1:x2]
            img = np.array(img)
            pred = predict_result(img)
        except:
            y1, x1, y2, x2 = box[0], box[1], box[2]+10, box[3]+10
            img = gray[y1:y2, x1:x2]
            img = np.array(img)
            pred = predict_result(img)
        finally:
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
            img = gray[y1:y2, x1:x2]
            img = np.array(img)
            pred = predict_result(img)
        print(img.shape)
        print(pred[0])
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # boxes[i].append(pred[0])
        ckd_word = check_word(pred[0])
        text.append(ckd_word)
        
    print(text)
    return text
    
if __name__ == "__main__":
    # img = Image.open("d:/projects/medical_app/server/zserver1/models/MLmodels/1 Adiflam.jpeg")
    # img = prepare_image(img)
    # predict_result(img)

    # labels = np.fromfile('d:/projects/medical_app/server/zserver1/models/MLmodels/test2.dat',dtype=float)
    # print(labels)
    image_path = "D:/projects/medical_app/MLs/object_detection/dataset/WhatsApp Image 2022-11-07 at 7.01.49 PM.jpeg"
    img = np.array(Image.open(image_path))
    predict_result_update(img)
    # print(check_word("cemethaon"))
