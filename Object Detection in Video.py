#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


config_file= 'C:\\Users\\LENOVO\\Desktop\\TensorFlow Object Detection API\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model= 'C:\\Users\\LENOVO\\Desktop\\TensorFlow Object Detection API\\ssd_mobilenet_v3_large_coco_2020_01_14\\ssd_mobilenet_v3_large_coco_2020_01_14\\frozen_inference_graph.pb'


# In[3]:


model= cv2.dnn_DetectionModel(frozen_model, config_file)


# In[ ]:





# In[4]:


classLabels= []
file_name= 'C:/Users/LENOVO/Desktop/TensorFlow Object Detection API/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[5]:


print(classLabels)


# In[6]:


print(len(classLabels))


# In[7]:


model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# Read an image

# In[8]:


img= cv2.imread('E:\PC Lovers\Bus.jpg')


# In[9]:


plt.imshow(img)  #bgr


# In[10]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[11]:


ClassIndex, confidence, bbox= model.detect(img, confThreshold= 0.5)


# In[12]:


print(ClassIndex)


# In[13]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[14]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[15]:


import cv2


# Video Demo

# In[16]:


cap= cv2.VideoCapture("E:\Desk\I- CLEAR Technologies Private Limited\Videos\sample1.mp4")


# check if the video is open correctly
if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
font_scale= 3
font= cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame= cap.read()
    
    ClassIndex, confidence, bbox= model.detect(frame, confThreshold= 0.55)
    
    print(ClassIndex)
    if (len(ClassIndex)!= 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<= 80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale= font_scale, color=(0,255,0), thickness= 3)
    cv2.imshow('Object Detection Tutorial', frame)
    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# Webcam Demo

# In[18]:


cap= cv2.VideoCapture(1)


# check if the video is open correctly
if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
font_scale= 3
font= cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame= cap.read()
    
    ClassIndex, confidence, bbox= model.detect(frame, confThreshold= 0.55)
    
    print(ClassIndex)
    if (len(ClassIndex)!= 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<= 80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale= font_scale, color=(0,255,0), thickness= 3)
    cv2.imshow('Object Detection Tutorial', frame)
    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




