#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from keras.models import load_model
# from time import sleep
# from tensorflow.keras.utils import img_to_array
# # from keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np
# import tensorflow as tf
from keras.preprocessing.image import img_to_array
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np




# In[2]:


face_classifier=cv2.CascadeClassifier("haarcascades_models/haarcascade_frontalface_default.xml")


# In[4]:


age_interpreter = tf.lite.Interpreter(model_path="age_detection_model_50epochs_no_opt.tflite")
age_interpreter.allocate_tensors()


# In[5]:


gender_interpreter = tf.lite.Interpreter(model_path="gender_detection_model_50epochs_no_opt.tflite")
gender_interpreter.allocate_tensors()


# In[6]:


age_input_details = age_interpreter.get_input_details()
age_output_details = age_interpreter.get_output_details()


# In[7]:


gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()


# In[8]:


age_input_shape = age_input_details[0]['shape']
gender_input_shape = gender_input_details[0]['shape']


# In[9]:


gender_labels = ['Male', 'Female']


# In[10]:


cap=cv2.VideoCapture(0)


# In[12]:


while True:
    ret,frame=cap.read()
    labels=[]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       
        #Gender
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        roi_color = np.array(roi_color).reshape(-1,200,200,3) #input shape is (1, 200,200,3)
        roi_color = roi_color.astype(np.float32)
        
        gender_interpreter.set_tensor(gender_input_details[0]['index'], roi_color)
        gender_interpreter.invoke()
        gender_preds = gender_interpreter.get_tensor(gender_output_details[0]['index'])
        
        
        #gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_preds>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50) #50 pixels below to move the label outside the face
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        
        age_interpreter.set_tensor(age_input_details[0]['index'], roi_color)
        age_interpreter.invoke()
        age_preds = age_interpreter.get_tensor(age_output_details[0]['index'])
        
        #age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_preds[0,0])
        age_label_position=(x+h,y+h)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    
    
    cv2.imshow('Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




