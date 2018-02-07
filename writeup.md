# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/hist.png "Visualization"
[image2]: ./images/preprocessed.png "Grayscaling"
[image3]: ./images/original.png "Original"
[image4]: ./md_test/Danger.jpg "Traffic Sign 1"
[image5]: ./md_test/Aniimal.jpg "Traffic Sign 2"
[image6]: ./md_test/Priority.jpg "Traffic Sign 3"
[image7]: ./md_test/Right.jpg "Traffic Sign 4"
[image8]: ./md_test/Straight.jpg "Traffic Sign 5"
[image9]: ./md_test/Yield.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for different classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Grayscale conversation, image data normalization, and histogram equalization preprocessing were used as preprocessing techniques.

* For grayscale conversation, I immediately used it after realizing the color may not an important factor and good results gained from the first project. I didn’t have the opportunity to compare its effectiveness since I had the grayscale preprocessing at the beginning.

* For image data normalization, I followed the project instruction: “For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project“, the image is normalized between [-1, 1] because the image range [0, 255]. I kept trying to improve the test accuracy for days by improving CNN settings but never considered to change the normalization range until I noticed “Minimally, the image data should be normalized so that the data has mean zero and equal variance”, which made me feel the normalization can play a big role to affect the training effect. Instead of [-1, 1], I used (b - a) * ( (img - 0) / (255 - 0) ) + a with a = 0.1 and b = 0.9 to normalize the image data between [0.1, 0.9]. The test actuary immediately changed from 0.89 to 0.96.

* For histogram equalization, due to the provided training data class varies as shown in the data histogram plot, we can generate additional more training images by rotating images by small angles and make each class has almost the same minimum number of training images. This preprocessing was proven to be optional for this project because the test accuracy improvement is negligible (from 0.961 to 0.962). I guess the reason is by simply generating additional training images by small rotation can’t provide extra useful features for the NN.

Summary: “Minimally, the image data should be normalized so that the data has mean zero and equal variance. Other pre-processing steps are optional. You can try different techniques to see if it improves performance.” Listen to the project instructions SERIOUSLY!

![alt text][image2]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



The final model used for training is LeNet1 with the following layers:

| Layer         		|     Description	        					| Input |Output|
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, ReLU activation 	|32x32x1 |28x28x32|
| Max pooling			| 2x2 stride				        		        |28x28x32|14x14x32|
| Convolution 5x5 	    | 1x1 stride, valid padding, ReLU activation 	|14x14x32|10x10x96|
| Max pooling			| 2x2 stride              	   					|10x10x96|5x5x96|
| Flatten				| 3 dimensions -> 1 dimension					|5x5x96| 2400|
| Fully Connected | ReLU activation, Dropout with keep_prob=0.5 to prevent overfitting 	|2400|600|
| Fully Connected | ReLU activation, Dropout with keep_prob=0.5 to prevent overfitting 	|600|150|
| Fully Connected | output = number of traffic signs   	|150| 43|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Adam Optimizer with a batch size of 128, 100 epochs, with a learning rate of 0.001 were used to obtain the results


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used three CNNs to conduct the training: LeNet, LeNet1 and LeNet2. The LeNet and LeNet1 are the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet2 used the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) architecture. The LeNet has exactly the same architecture as the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet used the same architecture with deeper filter size.

For LeNet3, as stated in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): "Usual ConvNets ([LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)) are organized in strict feed-forward layered architectures in which the output of one layer is fed only to the layer above." Contrary to the traditional LeNet, only the output of the second stage is fed to the classifier, in the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the output of the first stage is also branched out and fed to the classifier. By adding the procedure, the training yielded higher accuracies than the traditional method.

The input of all the tested networks is 32x32x1 image and the output is the probability of the 43 possible traffic signs.

Here are the training results for 3 models used:

* LeNet: rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.937, the testing accuracy is 0.922.

* LeNet1: rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.966, the test accuracy is 0.961.

* LeNet2, rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.962, the test accuracy is 0.950.

Summary: The improved (LeNet2) CNN has a better training result compared to the traditional LeNet (LeNet) using similar layer settings. In the same CNN architecture, deeper filter size results in better training result but more computational power.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4 ]![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

* The first image might be difficult to classify because of the curve
* The second image might be difficult to classify because of the the background and complex shape in the center
* The third image might be difficult to classify because of the different color than standard
* The fourth image might be difficult to classify because of the shape
* The fifth image might be difficult to classify because of the different color and shape
* The sixth image might be difficult to classify because of no shape in the center


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I downloaded six traffic sign images online to test the trained NN. Even all of them fall in the training data category, the trained NN never "see" them before.

| Image			        |     Prediction		|
|:---------------------:|:---------------------:|
| Dangerous curve to the left  | Dangerous curve to the left  |
| Go straight or left  		| Go straight or left 	|
| Priority road			| Priority road					|
| Right-of-way at the next intersection		| Right-of-way at the next intersection					|
| Wild animals crossing		| Wild animals crossing  |
| Yield | Yield |

6 of 6 correct = 100%

Compared with trained LeNet1 test accuracy 0.961, the new images test accuracy showed no surprise. This is due to the fact the new tested images are all in the training category and I had a well trained NN.

In the softmax probabilities, it shows the NN is very confident with its prediction (100%) with no prediction false. This may because the newly tested traffic signs are different enough to the second and third guess. I didn't test the signs isn't in the training set, because it's meaningless for the trained NN to give any predictive result. No matter how confident the predictive results are, they are all wrong.


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.



