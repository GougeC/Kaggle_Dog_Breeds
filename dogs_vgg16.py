import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import Model
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
def prepare_train_validation():
    image_size = 224
    labels = pd.read_csv('data/labels.csv.zip',compression='zip')
    #converting breed categories to categorical booleans
    one_hot_labels = pd.get_dummies(labels['breed']).values
    X_raw = []
    y_raw = []

    #loads and processes the images into image_size by image_size by 3 tensors
    for ind, row in enumerate(labels.values):
        img = cv2.imread('data/train/{}.jpg'.format(row[0]))
        img = cv2.resize(img,(image_size,image_size))
        X_raw.append(img)
        y_raw.append(one_hot_labels[ind,:])

    X = np.array(X_raw)
    y =np.array(y_raw)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2,stratify = y, random_state=1)
    mean_pixel = np.array([103.939, 116.779, 123.68])
    X_train = X_train - mean_pixel
    X_validation = X_validation - mean_pixel
    return X_train, X_validation, y_train, y_validation
X_train, X_validation, y_train, y_validation = prepare_train_validation()
image_size = 224
vgg_16_conv = VGG16(weights='imagenet',include_top = False)
img_in = Input(shape = (image_size,image_size,3),name = 'image_input')
outputVGG16 = vgg_16_conv(img_in)
X = Flatten()(outputVGG16)
X = Dense(2048, activation='relu',name="dense1")(X)
X = Dense(2048, activation='relu',name="dense2")(X)
X = Dense(120, activation='softmax', name="output")(X)
adapted_VGG16 = Model(input = img_in,output = X)
Ad = Adam(lr = .01 , beta_1 = .9, beta_2 = .999, epsilon = 1e-08, decay =0.0)
adapted_VGG16.compile(optimizer =Ad, loss = 'categorical_crossentropy',metrics = ['accuracy'] )

adapted_VGG16.fit(X_train,y_train,epochs = 10, batch_size = 19)
