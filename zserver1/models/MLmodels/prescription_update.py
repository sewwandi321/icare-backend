import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

PATH_TO_MODEL_DIR = "models/MLmodels/prescription_txt_loc"
PATH_TO_LABELS = "models/MLmodels/labels.pbtxt"

import time
from models.MLmodels.object_detection.utils import label_map_util
from models.MLmodels.object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def draw_my_vals(image,boxes,scores):
    threshhold = 0.30
    width, height, _ = image.shape
    values = []
    # for corrdinates in boxes:
    # print(len(boxes))
    # print(len(scores))
    for i in range(len(boxes)):
        if scores[i] >= threshhold:
            y,x,y1,x1 = boxes[i][0]*width, boxes[i][1]*height, boxes[i][2]*width, boxes[i][3]*height
            values.append([int(y),int(x),int(y1),int(x1)])
    #         cv2.rectangle(image, (int(x), int(y)), (int(x1), int(y1)), (10, 255, 0), 2)
    #         cv2.putText(image, str(scores[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # cv2.imshow("image",image)
    
    
    return values

def predict(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    values = draw_my_vals(image_np,detections['detection_boxes'],detections['detection_scores'])
    
    return values

    
def main():
    # print('Running inference for {}... '.format(image_path), end='')
    image_path = "D:/projects/medical_app/MLs/object_detection/dataset/WhatsApp Image 2022-11-07 at 6.13.22 PM.jpeg"
    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    print('Done')
    values = draw_my_vals(image_np,detections['detection_boxes'],detections['detection_scores'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return values
    

if __name__ == '__main__':
    main()
