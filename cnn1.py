# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:57:18 2020

@author: HSU6KOR
"""
#part1 Building CNN
#Building CNN model to classify Dogs and Cats

#importing the keras lib and pckages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize CNN using Sequential package
classifier = Sequential()
#adding different layers
# 1st layer is CNN
#Step 1 = convolution
classifier.add(Convolution2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
#Step 2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 fully connection
#hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
#output layer 
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2
#Image preprocessing step, fitting above CNN model to images
from keras.preprocessing.image import ImageDataGenerator
#image agumentation for traingin set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#preprocess test_set
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


