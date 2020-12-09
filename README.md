# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/nn.svg "Model Visualization"
[image2]: ./output_images/center_2016_12_01_13_44_46_993.jpg "Center Lane Driving"
[image3]: ./output_images/original.png "Original Image"
[image4]: ./output_images/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 containing a recorded video in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The chosen model is a LeNet-5 architecture.

In detail, both convolutional layers have 5x5 filter sizes and depths of 6 respectively 16 (model.py lines 92 + 96) 

The model includes RELU layers to introduce nonlinearity. However, the ReLU layers are only included after the convolutions (code lines 92 + 96), not after the fully connected layers because the network is a regression network predicting the steering angle. The data is normalized in the model using a Keras lambda layer (code line 88) and cropped to exclude the sky and the hood of the car (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. These layers are inserted after the fully connected layers (model.py lines 102 + 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The split into training and validation set appears in code line 29. Creation of generators for training and validation samples is carried out in code lines 75 + 76. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an SGD optimizer with momentum for minimizing the mean squared error loss function. The learning rate was tuned manually. Its value was set to 0.0001 while the momentum was fixed at 0.9 (model.py line 109).

#### 4. Appropriate training data

Since my own collected training data didn't induce a good driving behavior in autonomous mode I took the prerecorded sample driving data. Based on these data the vehicle could be successfully kept on the road. However, I haven't got any insight into how the training data got acquired, e.g. how many laps of center lane driving, recovery from the left and right sides of the road, driving counter-clockwise or driving smoothly around curves are included. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

During the experimental process of finding a well performing solution I was sticking to the LeNet-5 architecture.

At the beginning I used an Adam optimizer minimizing the mean squared error loss and chose a learning rate of 0.001. My final solution uses a SGD optimizer with a learning rate of 0.0001 and momentum of 0.9. During the process of finding good hyperparameters I made sure to keep overfitting at a minimum level and that the model learns from the provided data. In total, training was carried out for 250 epochs.

Based on my collected training data the car drove safely around track one right until it had crossed the bridge. Instead of staying on the real road it takes the side path on the right which might be a consequence of the data collection process.

In detail, I collected 2 laps of center lane driving around track one, 1 lap of counter clockwise center lane driving around track one and 1 lap focusing on driving smoothly around curves of track two.

Because of the car taking the side path mentioned above I took the prerecorded sample driving data for training the model. Finally, the car could be kept successfully on the road throughout all of track one in autonomous mode.

#### 2. Final Model Architecture

As already mentioned the chosen architecture is a LeNet-5 architecture with dropout layers included after the fully connected layers.

Here is a visualization of the architecture. Note that in this picture the dropout layers are missing.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Since I finally chose the prerecorded sample driving data I cannot make any statements about the detailed collection process of the dataset.

However, I guess that the following image is an example image of center lane driving:

![alt text][image2]

In total, the prerecorded sample drving data have got 8036 data points. After random shuffling of data 80% build the training set while 20% are held out for the validation set. Each of these data got preprocessed within their corresponding generator (on the fly) by the following two techniques.

In order to consider recovery driving from the sides of the road I manually added/subtracted a small correction factor (0.2) to the steering measurement for each left respectively right image during training. By this way the model learns to steer softer/harder, especially when curves appear. Moreover, by this approach more data points are considered for training helping the model to generalize better.

Furthermore each sample was augmented by flipping the image and taking the opposite angle. I thought that this could avoid biased steering angle predictions because the curves of track one are mainly left curves. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The  number of epochs used for training was 250. As already mentioned, I used an SGD optimizer with momentum with the learning rate being set to 0.0001.
