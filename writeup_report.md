#**Behavioral Cloning** 


###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the NVIDIA convolution neural network, which contains:
1. 5x5 filter with depth of 24 and RELU activation
2. 5x5 filter with depth of 36 and RELU activation
3. 5x5 filter with depth of 48 and RELU activation
4. 3x3 filter with depth of 64 and RELU activation
5. 3x3 filter with depth of 64 and RELU activation
6. Fully connected layer with 100 outputs
7. Fully connected layer with 50 outputs
8. Final layer with 1 output

The input data is also normalized and cropped to speed up the training process.

####2. Attempts to reduce overfitting in the model

I added dropout layer after each of the two 3x3 convolution layers.

I also trained the model with both udacity data and my own driving data sets, including one lap driving in the reverse direction, which helped reduce the overfitting (model.py lines 27-40).


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 158).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also one lap driving in the reverse direction.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse the NVIDIA model as much as possible and supply various good quality training data.

My first step was to build the NVIDIA convolution neural network model. After the initial test with just one set of driving data and 3 epochs, the error was down to 0.019, which implied that this model should be a good fit for this project.

To further test the model, I collected more data and split it into a training and validation set. I found that when training for 5 epochs, my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I reduced the number of epochs to 3, and collected more data, such as "very smooth driving around the curves," "recovery driving," "driving in the reverse direction," etc. I also used udacity data together with my own driving data.

Then I trained the model, it turned out that there was too much oscillation in lateral direction. I thought it might be related to the recovery driving data where I introduced too much sharp turn maneuvers which affected the model.

Therefore in the final step, I removed the recovery driving data set and trained the model with 3 epochs. The model turned out pretty good. It was able to drive the car for multiple laps without falling off the track or even touch the lane markers.


####2. Final Model Architecture

The final model architecture (model.py lines 142-157) is the same as the NVIDIA model that consisted of a convolution neural network with the following layers and layer sizes:
1. 5x5 filter with depth of 24 and RELU activation
2. 5x5 filter with depth of 36 and RELU activation
3. 5x5 filter with depth of 48 and RELU activation
4. 3x3 filter with depth of 64 and RELU activation
5. dropout layer - 0.3
6. 3x3 filter with depth of 64 and RELU activation
7. dropout layer - 0.3
8. Fully connected layer with 100 outputs
9. Fully connected layer with 50 outputs
10. Final layer with 1 output

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

Here is an example image of center lane driving:
![Image Cropping][./writeup_images/before_and_after_crop.jpg]

I then recorded another two laps driving in the center of the lane.

Later I drove 1 lap using recovery driving.

I later added another 4 sets of center lane driving data.

To augment the data sat, I drove another two laps in the reverse direction.

I also flipped the images to expand the training set and reduce overfitting to couterclockwise driving.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as the validation error will go up compared to training error if 5 epochs was used.

I eventually used an adam optimizer so that manually training the learning rate wasn't necessary.
