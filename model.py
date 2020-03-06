#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:22:03 2020

@author: silvio
"""


# loading the data
import csv
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import math

# load samples
samples = []
with open('./data_prerecorded/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)
        

# perform a split into training and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# generate data for training (on the fly) 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data_prerecorded/data/IMG/' + filename
                    image = ndimage.imread(current_path)
                    images.append(image)
                    
                    # data augmentation
                    images.append(np.fliplr(image))
                
                measurement = float(batch_sample[3])
    
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = measurement + correction
                steering_right = measurement - correction
                
                # considering also the opposite sign of steering angles for the augmented images
                angles.extend([measurement,measurement*-1.0,steering_left,steering_left*-1.0,steering_right,steering_right*-1.0])
                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)    
 


# Set our batch size
batch_size=32


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



# building and training the network 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from tensorflow.keras import optimizers
       

#LeNet-5 regression network 
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
print(model.layers[-1].output_shape)
model.add(Cropping2D(cropping=((70,25),(0,0))))
print(model.layers[-1].output_shape)
model.add(Conv2D(6,(5,5),activation="relu"))
print(model.layers[-1].output_shape)
model.add(MaxPooling2D())
print(model.layers[-1].output_shape)
model.add(Conv2D(16,(5,5),activation="relu"))
print(model.layers[-1].output_shape)
model.add(MaxPooling2D())
print(model.layers[-1].output_shape)
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))


# configuring the model for training
model.compile(optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),loss='mse')

# training the model on data generated batch-by-batch by a generator
model.fit_generator(train_generator,\
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size),\
                    validation_data=validation_generator,\
                    validation_steps=math.ceil(len(validation_samples)/batch_size),\
                    epochs=250, verbose=1)

# save the trained model
model.save('model.h5')