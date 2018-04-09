# Dog Breed Identification

This repository is my work for the [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) competition from Kaggle.

My strategy for this competition is to use CNN pretrained on the imagenet dataset as feature extraction, then train a small neural network using these features to classify each image into one of 120 breeds. I also included my code to reorganize the training set that they give into directories based on breed so that I could use the Keras [ImageDataGenerator](https://keras.io/preprocessing/image/) tool. 

The process currently outlined in the notebook is my experiment on deciding between a network with global max pooling after the pretrained layers and a network without that pooling layer, but with regularization in the form of dropout. Before the experiment listed out in the notebook I did play around with some different pretrained base networks such as VGG19 and VGG16 and I also tried out some different optimizers, but I did not include that in this notebook.

The results of this model are currently about 92 - 95% top 5 accuracy with about 70% accuracy in predicting the exact breed in the image. My best model has so far been the pretrained Inception V3 network with dropout .3 before two fully connected layers and then the output layer. The next step for this project on my todo list is to run an organized experiment on how big these two fully connected layers should be, as right now they are fairly arbitrarily sized. Additionally I would like to experiment with fine tuning the last convolutional blocks in the base layers so that the features extracted can be more relavent to this specific problem. 
