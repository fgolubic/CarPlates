'''
Created on 4. pro 2017.

@author: Filip
'''
from keras.preprocessing.image import img_to_array
import cv2
import numpy
from Networks import LeNet5, CustomNet, AlexNet

model = LeNet5(True)
#model= CustomNet(True)
#model=AlexNet(True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',

              metrics=['accuracy'])



imgs = ['../symbols/4.jpg', '../symbols/7.jpg', '../symbols/8.jpg', '../symbols/d.jpg', '../symbols/g.png', 
        '../symbols/h.jpg', '../symbols/u.jpg', '../symbols/z.png']
characters = '0123456789ABCDEFGHIJKLMNOPRSTUVZÄ†ÄŚÄ�Ĺ Ĺ˝'

for filename in imgs:
    img = cv2.imread(filename)
    img = cv2.resize(img, (24, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = img.astype("float") / 255.0
    img2 = img_to_array(img2)
    img2 = numpy.expand_dims(img2, axis=0)
    preds = model.predict(img2)
    print("Image: ", filename, " - classified as: ", characters[numpy.argmax(preds[0])])
    print("Position of maximum value =", numpy.argmax(preds[0]), "; maximum value = ", numpy.max(preds[0]))



