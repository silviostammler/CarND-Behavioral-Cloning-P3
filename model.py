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

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    # recovery driving considering center image, left image and right image
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        
    measurement = float(line[3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = measurement + correction
    steering_right = measurement - correction
    
    measurements.extend([measurement,steering_left,steering_right])
    


#data augmentation
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    #plt.imshow(augmented_images[-1])
    #plt.show()
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    #plt.imshow(augmented_images[-1])
    #plt.show()
    augmented_measurements.append(measurement*-1.0)

    
X_train = np.array(images)
y_train = np.array(measurements)



# building and training the network 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D
       

#LeNet-5 regression network 
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(16,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')