######## Image Object Detection Using Tensorflow-trained Classifier #########
# Import packages
import os
import cv2
import imutils
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import time
from threading import Thread

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_1128'
images = ['pedxng.png, speedLimit30_1333396738.avi_image3.png, '
          'stop_1333397610.avi_image3.png, '
          'signalAhead_1331866722.avi_image5.png']

# Grab path to current working directory
CWD_PATH = os.getcwd()
tf.disable_v2_behavior()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 40

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def GetClassName(data):
    for cl in data:
        print(cl)
        return cl['name']
        
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

for i in images:
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(os.path.join("C:\\Users\\Hasan\\models\\research\\object_detection",i))
    image = imutils.resize (image, width=640)
    print(image.shape)
    image_expanded = np.expand_dims(image, axis=0)
    start_time = time.time()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    end_time = time.time()
    # Draw the results of the detection (aka 'visulaize the results')

    print("Inference took {} seconds ".format(end_time - start_time))
    print (boxes, scores, classes, num)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.80)

    #data processed
    data = [category_index.get(value) for index,value in enumerate(classes[0]) if
            scores[0,index] > 0.8]

    print (GetClassName (data))
    # Play Music on Separate Thread (in background)
    music_thread = Thread (target=text2speech (GetClassName (data)))
    music_thread.start ()

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)
# Clean up
cv2.destroyAllWindows()
