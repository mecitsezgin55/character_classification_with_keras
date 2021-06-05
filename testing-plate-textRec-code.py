# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:35:24 2021

@author: mecit.sezgin
"""

import numpy as np
import cv2
from keras.models import load_model



########### PARAMETERS ##############
class_names = ["0","1","2","3","4","5","6","7","8","9","A",
               "B","C","D","E","F","G","H","I","J","K","L",
               "M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
#####################################
 
#### LOAD THE TRAINNED MODEL
print("model yükleniyor")
model = load_model("character_model.h5")
print("model yükendi")
 
#### PREPORCESSING FUNCTION
def preProcessing(image):

    image = np.asarray(image)
    image = cv2.resize(image,(32,55))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    image = cv2.erode(image, rectKern, iterations=1)
    image = cv2.dilate(image, rectKern, iterations=1)
    
    image = cv2.equalizeHist(image)
    image = image/255
    return image

#### PREDICT
def predictText(image):
    image = image.reshape(1,55,32,1)
    classIndex = int(model.predict_classes(image))
    #print(classIndex)
    predictions = model.predict(image)
    #print(predictions)
    probVal= np.amax(predictions)
 
    return class_names[classIndex],probVal
    

# resmi yukle
imgOriginal = cv2.imread("foto/M4.png")
cv2.imshow("Original Image",imgOriginal)

#resmi siniflandirmak için uygun formata getir
img = preProcessing(imgOriginal)

#tahmin et
pre, prob = predictText(img)
print(pre,prob)

#cv2.putText(img,str(pre) + "   "+str(prob),(50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)
#cv2.imshow("Processsed Image",img)


cv2.waitKey(0)
cv2.destroyAllWindows()




