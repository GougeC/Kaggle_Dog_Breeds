#import relevant packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.applications import ResNet50
from keras import Model
import cv2
from sklearn.model_selection import train_test_split

#load the data
labels = pd.read_csv('../data/dogs/data/labels.csv.zip',compression='zip')
labels.head(2)

#The images are all different dimensions so this will be the parameter to
#resize them to be square
image_size = 128

def prepare_train_validation():
    #converting breed categories to categorical booleans
    one_hot_labels = pd.get_dummies(labels['breed']).values
    X_raw = []
    y_raw = []
    print('processing data...')
    #loads and processes the images into image_size by image_size by 3 tensors
    for ind, row in enumerate(labels.values):
        img = cv2.imread('../data/dogs/data/train/{}.jpg'.format(row[0]))
        img = cv2.resize(img,(image_size,image_size))
        X_raw.append(img)
        y_raw.append(one_hot_labels[ind,:])
    X = np.array(X_raw)
    y =np.array(y_raw)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2,stratify = y, random_state=1)
    return X_train, X_validation, y_train, y_validation

def conv_network_1( input_shape,weights_path = None):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=input_shape))

    model.add(Convolution2D(32,(3,3),activation ='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,(3,3),activation ='relu'))
    #model.add(MaxPooling2D((2,2),strides = (2,2)))

    model.add(Convolution2D(64,(3,3),activation ='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,(3,3),activation ='relu'))
    model.add(MaxPooling2D((2,2),strides = (2,2)))

    model.add(Convolution2D(128,(3,3),activation ='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,(3,3),activation ='relu'))
    model.add(MaxPooling2D((2,2),strides = (2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',name="dense1"))
    model.add(Dense(1024, activation='relu',name="dense2"))
    model.add(Dense(120, activation= 'softmax',name="dense3"))
    return model
X_train, X_validation, y_train, y_validation = prepare_train_validation()
model = conv_network_1(input_shape = (image_size,image_size,3))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'] )
model.fit(X_train,y_train,epochs = 10, batch_size = 19)
