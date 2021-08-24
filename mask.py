import cv2
import numpy as np
import time
import argparse
import os
import config
from mylib import config
from pygame import mixer

#Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
    
print(cv2.__version__)
labelsPath_mask = os.path.sep.join([config.MODEL_PATH, "obj.names"])
LABELS_mask = open(labelsPath_mask).read().strip().split("\n")

import cv2
count = cv2.cuda.getCudaEnabledDeviceCount()
print("CUDA ENABLED DEVICES: ")
print(count)

COLORS = [[0, 0, 255], [0, 255, 0],[0, 0, 255],[0, 0, 255]]

#derive the paths to the YOLO weights and model configuration
weightsPath_mask = os.path.sep.join([config.MODEL_PATH, "2021s1_face_mask.weights"])
configPath_mask = os.path.sep.join([config.MODEL_PATH, "2021s1_face_mask.cfg"])

#load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net_mask = cv2.dnn.readNetFromDarknet(configPath_mask, weightsPath_mask)

print("[INFO] setting preferable backend and target to CUDA...")
net_mask.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_mask.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#determine only the *output* layer names that we need from YOLO
layer_names = net_mask.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net_mask.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None
writer = None

def alert():
    mixer.init()
    alert = mixer.Sound('beep-07.wav')
    alert.play()
    time.sleep(0.1)
    alert.play()


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#initiate variables for calculating fps
new_frame_time=0
prev_frame_time=0

while True:
    _, frame = cap.read()

    height, width, channels = frame.shape

    # Detecting objects
    blob_mask = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net_mask.setInput(blob_mask)
    outs_mask = net_mask.forward(output_layers)
    print(outs_mask)

    # Showing informations on the screen
    class_ids = []
    confidences_mask = []
    boxes_mask = []
    for out in outs_mask:
        for detection_mask in out:
            scores_mask = detection_mask[5:]
            class_id = np.argmax(scores_mask)
            confidence = scores_mask[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection_mask[0] * width)
                center_y = int(detection_mask[1] * height)
                w = int(detection_mask[2] * width)
                h = int(detection_mask[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes_mask.append([x, y, w, h])
                confidences_mask.append(float(confidence))
                class_ids.append(class_id)

    indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.8, 0.3)

    if len(indexes_mask) > 0:
        # loop over the indexes we are keeping
        for i_mask in indexes_mask.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes_mask[i_mask][0], boxes_mask[i_mask][1] )
            (w, h) = (boxes_mask[i_mask][2], boxes_mask[i_mask][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_ids[i_mask]]]
            if [0, 0, 255] == color:
                #alert()
                print("violation")
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(LABELS_mask[class_ids[i_mask]], confidences_mask[i_mask])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    #Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps=str(int(fps))
    #Put FPS Counter on image
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
