#Dog Breed Identification

This repository is my work for the [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) competition from Kaggle.

My strategy for this competition is to use CNN pretrained on the imagenet dataset as feature extraction, then train a small neural network using these features to classify each image into one of 120 breeds. I also included my code to reorganize the training set that they give into directories based on breed so that I could use the Keras [ImageDataGenerator](https://keras.io/preprocessing/image/) tool 
