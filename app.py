
import numpy as np
import cv2
import pyttsx3
engine = pyttsx3.init()

thres = 0.5 # Threshold to detect object
#img = cv2.imread("sherlock.jpg")
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
#print(classNames)



weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    success,img = cap.read()
    classIDs,confs,bbox = net.detect(img,confThreshold=thres)
    if len(classIDs) != 0:
        for classID,confidenece,box in zip(classIDs.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color = (255,0,0),thickness = 3)
            cv2.putText(img,classNames[classID-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
            engine.say(classNames[classID-1])
            engine.runAndWait()
            engine.stop()

    cv2.imshow("Output",img)
    cv2.waitKey(1)
