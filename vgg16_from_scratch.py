import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import Model
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
def prepare_train_validation():
    image_size = 224
    labels = pd.read_csv('data/data/labels.csv.zip',compression='zip')
    #converting breed categories to categorical booleans
    one_hot_labels = pd.get_dummies(labels['breed']).values
    X_raw = []
    y_raw = []

    #loads and processes the images into image_size by image_size by 3 tensors
    for ind, row in enumerate(labels.values):
        img = cv2.imread('data/data/train/{}.jpg'.format(row[0]))
        img = cv2.resize(img,(image_size,image_size))
        X_raw.append(img)
        y_raw.append(one_hot_labels[ind,:])
    X = np.array(X_raw)
    y =np.array(y_raw)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2,stratify = y, random_state=1)
    mean_pixel = np.mean(X_train,axis = (0,1,2))
    X_train = X_train - mean_pixel
    X_validation = X_validation - mean_pixel
    return X_train, X_validation, y_train, y_validation
X_train, X_validation, y_train, y_validation = prepare_train_validation()
image_size = 224
vgg_16_conv = VGG16(weights=None,include_top = True,classes = 120)
img_in = Input(shape = (image_size,image_size,3),name = 'image_input')
outputVGG16 = vgg_16_conv(img_in)
adapted_VGG16 = Model(input = img_in,output = outputVGG16)
Ad = Adam(lr = .1 , beta_1 = .9, beta_2 = .999, epsilon = 1e-08, decay =0.0)
sgd = SGD(lr = .01 , momentum = 0.0, decay=  0.000010)
adapted_VGG16.compile(optimizer =sgd, loss = 'categorical_crossentropy',metrics = ['accuracy'] )

adapted_VGG16.fit(X_train,y_train,epochs = 1000, batch_size = 19)
