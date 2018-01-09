#import relevant packages
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import Model
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#The images are all different dimensions so this will be the parameter to
#resize them to be square
img_size = 112

#load the data
labels = pd.read_csv('~/dogs/Kaggle_Dog_Breeds/data/data/labels.csv.zip',compression='zip')
t_labels = pd.read_csv('~dogs/Kaggle_Dog_Breeds/data/sample_submission.csv')
labels.head(2)

targets_series = pd.Series(labels['breed'])

one_hot = pd.get_dummies(targets_series,sparse = True)

oh_labels = np.asarray(one_hot)
x_train = []
y_train = []
x_test = []

i = 0
for path, breed in tqdm(labels.values):
    img = cv2.imread('~dogs/Kaggle_Dog_Breeds/data/data/train/{}.jpg'.format(path))
    label = oh_labels[i]
    x_train.append(cv2.resize(img,(img_size,img_size)))
    y_train.append(label)
    i+=1
for path in tqdm(t_labels['id'].values):
    img = cv2.imread('~/dogs/Kaggle_Dog_Breeds/data/test/{}.jpg'.format(path))
    x_test.append(cv2.resize(img,(img_size,img_size)))


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.

number_classes = y_train_raw.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=1)
base_model = VGG16(weights = 'imagenet', include_top = False,
                      input_shape = (img_size,img_size,3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(number_classes, activation = 'softmax')(x)
model = Model(inputs= base_model.input,outputs = predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor = 'val_acc',patience = 3,verbose = 1)]

model.fit(X_train,Y_train,epochs = 8, validation_data= (X_valid,Y_valid),verbose = 1)

preds = model.predict(x_test,verbose = 1)

sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0,'id',t_labels['id'])
sub.head()

sub.to_csv('output1.csv')
