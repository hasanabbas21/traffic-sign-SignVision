######## Video Object Detection Using Tensorflow-trained Classifier #########

# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import sys
from threading import Thread
import time
import imutils
import pyttsx3

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_1128'
VIDEO_NAME = 'NO20200112-025953-000237F.mov'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 40

def GetClassName(data):
    for cl in data:
        return cl['name']

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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

# Open video file
# video = cv2.VideoCapture('C:\\Users\\Hasan\\models\\research\\object_detection\\traffic scenes\\merge and end lane - Test 1.mov')
# video = cv2.VideoCapture('C:\\Users\\Hasan\\models\\research\\object_detection\\traffic scenes\\ped crossing.mov')
# video = cv2.VideoCapture('C:\\Users\\Hasan\\models\\research\\object_detection\\traffic scenes\\school -1.mov')
# video = cv2.VideoCapture('C:\\Users\\Hasan\\models\\research\\object_detection\\traffic scenes\\stop -1 .mov') 
video = cv2.VideoCapture('C:\\Users\\Hasan\\models\\research\\object_detection\\traffic scenes\\combined.mov') 

fps = video.get (cv2.CAP_PROP_FPS)
video.set(cv2.CAP_PROP_FPS, 5)
print ("---" ,fps,  " ---")
video.set(3,640)
video.set(4,480)
time.sleep(5)

n = 0
sign_list = []
while(video.isOpened()):
    ret, frame = video.read ()
    # frame = imutils.resize (frame, width=640)
    frame = cv2.resize (frame, (640, 380))
    print(frame.shape)
    frame_expanded = np.expand_dims(frame, axis=0)
    video.set (cv2.CAP_PROP_POS_MSEC, n * 2000)
    n = n + .05

    start_time = time.time ()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    end_time = time.time ()

    print ("Inference took {} seconds ".format (end_time - start_time))
    print (boxes, scores, classes, num)
    # Draw the results of the detection (aka 'visulaize the results')
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.85)

    data = [category_index.get (value) for index, value in
            enumerate (classes[0]) if scores[0, index] > 0.9]
    sign_name = GetClassName (data)
    print (type(sign_name))

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
