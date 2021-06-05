

import numpy as np
import cv2
from keras.models import load_model


########### PARAMETERS ##############

class_names = ["0","1","2","3","4","5","6","7","8","9","A",
               "B","C","D","E","F","G","H","I","J","K","L",
               "M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
s=0
#####################################
 
#### LOAD THE TRAINNED MODEL
print("model yükleniyor")
model = load_model("character_model.h5")
print("model yüklendi")


def findCharacter(image): 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    _, contours,hierarch = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    contour_gray = gray.copy()
    
     # sayilarin ici dolduruluyor
    for i in range(len(contours)):
        cv2.drawContours(contour_gray,contours,i,255,-1)
    
#    cv2.imshow("contour_gray",contour_gray)
    
    #tekrar contour lar bulunuyor
    _, contours,hierarch = cv2.findContours(contour_gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    # contour lar x e göre siralaniyor
    contours = sorted(contours,key =cv2.boundingRect,reverse= False)
    #print(len(contours))
    
    tahmin = ""
    fg=0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c) 
        cv2.rectangle(contour_gray, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        if x != 0:
            x = x-1
        if y != 0:
            y = y-1
        w = w+1
        h = h+1
        
        
        #karakterlerin bulunduğu alan alınıyor
        bolge = gray[y:(y+h), x:(x+w)]

        #resim sınıflandırma için uygun şekle getiriliyor
        bolge = preProcessing(bolge)
                   
        #sınıflandırma yapılıyor
        karakterTahmin, karakterTahminYuzdesi = predictText(bolge)
       
        
        if karakterTahminYuzdesi > 0.6:
            tahmin += karakterTahmin
            
        print("krac:",karakterTahmin,"  yuzde:",karakterTahminYuzdesi," | ", "fg :",fg)
        cv2.imshow("bolgeler"+str(fg),bolge)
        fg +=1
    
    cv2.imshow("contour_gray2",contour_gray)
    return tahmin


 
#### PREPORCESSING FUNCTION
def preProcessing(image):
    image = np.asarray(image)
    image = cv2.resize(image,(32,55))
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
imgOriginal = cv2.imread("foto/asd.png")

karakterler = findCharacter(imgOriginal)
print(karakterler)

cv2.imshow("Original Image",imgOriginal)


cv2.waitKey(0)
cv2.destroyAllWindows()




