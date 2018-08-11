'''
Created on 8. sij 2018.

@author: Filip
'''
from keras.preprocessing.image import ImageDataGenerator
from Networks import CustomNet


model=CustomNet(False)



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



batch_size = 64
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        "../gen/training data",
        target_size=(32,24),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        "../gen/validation data",
        target_size=(32,24),
        batch_size=batch_size,
        class_mode='categorical')

# Ispis podataka o redoslijedu izlaza mreže
print(train_generator.class_indices)
print(train_generator.classes)
print(validation_generator.class_indices)
print(validation_generator.classes)


model.fit_generator(train_generator, 111000 // batch_size, validation_data = validation_generator, validation_steps = 11100 // batch_size, epochs =10)

model.save_weights('Custom.h5')

