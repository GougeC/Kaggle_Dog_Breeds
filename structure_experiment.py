import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import Model
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split


def prepare_train_validation(new_image_size):
    image_size = new_image_size
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

    return X_train, X_validation, y_train, y_validation

def train_eval_VGG_pretrained_weights(epochs,batch_size,optimizer,data):
    X_train, X_validation, y_train, y_validation = data
    image_size = X_train[1]
    vgg_16_conv = VGG16(weights='imagenet',include_top = False)
    img_in = Input(shape = (image_size,image_size,3),name = 'image_input')
    outputVGG16 = vgg_16_conv(img_in)
    X = Flatten()(outputVGG16)
    X = Dense(4096, activation='relu',name="dense1")(X)
    X = Dense(4096, activation='relu',name="dense2")(X)
    X = Dense(120, activation='softmax', name="output")(X)
    adapted_VGG16 = Model(input = img_in,output = X)
    adapted_VGG16.compile(optimizer = optimizer, loss = 'categorical_crossentropy',metrics = ['accuracy'] )
    history = adapted_VGG16.fit(X_train,y_train,epochs = epocs, batch_size = batch_size,verbose = 0)
    metrics = adapted_VGG16.evaluate(X_validation,y_validation)
    return history, metrics

def train_eval_VGG16(epochs,batch_size,optimizer,data):
    X_train, X_validation, y_train, y_validation = data
    image_size = X_train[1]
    vgg_16_conv = VGG16(weights=None,include_top = True,classes = 120)
    img_in = Input(shape = (image_size,image_size,3),name = 'image_input')
    outputVGG16 = vgg_16_conv(img_in)
    adapted_VGG16 = Model(input = img_in,output = outputVGG16)
    adapted_VGG16.compile(optimizer =optimizer, loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'] )

    history = adapted_VGG16.fit(X_train,y_train,epochs = epochs, batch_size = batch_size,verbose = 0)

    metrics = adapted_VGG16.evaluate(X_validation,y_validation)


def train_eval_ResNet50(epochs,batch_size,optimizer, data):
    X_train, X_validation, y_train, y_validation = data
    model = ResNet50(include_top=True, weights='imagenet', classes=120)
    history = model.fit(X_train,y_train,epochs = epochs,batch_size = batch_size,verbose = 0)
    metrics = model.evaluate(X_validation,y_validation)

if __name__ == "__main__":
    data = list(prepare_train_validation(224))
    mean_pixel = np.mean(data[0],axis = (0,1,2))
    mean_subtracted_data = data
    mean_subtracted_data[0]= data[0] - mean_pixel
    mean_subtracted_data[1]= data[1] - mean_pixel
    with open("results_1.txt", "w") as res_file:
        res_file.write("VGG16 without pretrained weights:")
        sgd = SGD(lr=0.01, decay=1e-6)
        h1,m1 = train_eval_VGG16(50,19,sgd,mean_subtracted_data)
        res_file.write(m1)
        res_file.write(h1)
        res_file.write("VGG16 with pretrained weights:")
        sgd = SGD(lr=0.01, decay=1e-6)
        h2,m2 = train_eval_VGG_pretrained_weights(50,19,sgd,mean_subtracted_data)
        res_file.write(m2)
        res_file.write(h2)
        res_file.write("ResNet50 pretrained:")
        h3,m3train_eval_ResNet50(50,19,sgd,data)
        res_file.write(m3)
        res_file.write(h3)
