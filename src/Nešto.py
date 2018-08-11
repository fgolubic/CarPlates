'''
Created on 26. sij 2018.

@author: Filip
'''


from keras.preprocessing.image import img_to_array
import cv2
import numpy
from Networks import LeNet5
from PIL import Image
import os

def Run():

    model = LeNet5(True)

    imgs= os.listdir('../TableDetection/plate/')
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


    characters = '0123456789ABCDEFGHIJKLMNOPRSTUVZÄ†ÄŚÄ�Ĺ Ĺ˝'

    for im in imgs:
        img = cv2.imread(im)   
        img = cv2.resize(img, (24, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = img.astype("float") / 255.0
        img2 = img_to_array(img2)
        img2 = numpy.expand_dims(img2, axis=0)
        preds = model.predict(img2)
        print("Image: ",  " - classified as: ", characters[numpy.argmax(preds[0])])
        print("Position of maximum value =", numpy.argmax(preds[0]), "; maximum value = ", numpy.max(preds[0]))



