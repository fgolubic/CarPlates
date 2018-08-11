'''
Created on 26. sij 2018.

@author: Filip
'''


from keras.models import Sequential
from keras.layers import Dense, Dropout,  Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D




def LeNet5(isTest):
    num_filters = 16
    filter_size = 5
    model=Sequential()
    model.add(Convolution2D(32, (filter_size, filter_size), border_mode='same', activation='relu', input_shape=(32,24,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(num_filters, (filter_size, filter_size), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(num_filters, (filter_size, filter_size), activation='relu', border_mode='same'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(37, activation='softmax'))
    
    
    if isTest is True:
        model.load_weights('../weights/LeNet5.h5')
    
    return model



def AlexNet(isTest):
    model=Sequential()
    model.add(Convolution2D(32, (5,5), activation='relu', input_shape=(32,24,3)))
    model.add(Convolution2D(32, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(Convolution2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense( 37, activation='softmax'))
    
    
    if isTest is True:
        model.load_weights('../weights/alexNet.h5')
        
        
    return model
    
    
    
    
def CustomNet(isTest):
    model=Sequential()
    
    model.add(Convolution2D(32, (16, 16), border_mode='same', activation='relu', input_shape=(32,24,3)))
    model.add(Convolution2D(10, (16, 16), border_mode='same', activation='relu', input_shape=(32,24,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(5, (16, 16), activation='relu', border_mode='same'))
    model.add(Convolution2D(5, (16, 16), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(10, (16, 16), activation='relu', border_mode='same'))
    model.add(Convolution2D(5, (8, 8), activation='relu', border_mode='same'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(37, activation='softmax'))
    
    
    if isTest is True:
        model.load_weights('../weights/Custom.h5')
    
    return model