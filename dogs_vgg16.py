def prepare_train_validation():
    #converting breed categories to categorical booleans
    one_hot_labels = pd.get_dummies(labels['breed']).values
    X_raw = []
    y_raw = []
    #loads and processes the images into image_size by image_size by 3 tensors
    for ind, row in enumerate(labels.values):
        labels = pd.read_csv('../data/dogs/data/labels.csv.zip',compression='zip')
        img = cv2.imread('../data/dogs/data/train/{}.jpg'.format(row[0]))
        img = cv2.resize(img,(image_size,image_size))
        X_raw.append(img)
        y_raw.append(one_hot_labels[ind,:])
    X = np.array(X_raw)
    y =np.array(y_raw)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2,stratify = y, random_state=1)

    return X_train, X_validation, y_train, y_validation
X_train, X_validation, y_train, y_validation = prepare_train_validation()

vgg_16_conv = VGG16(weights='imagenet',include_top = False)
img_in = Input(shape = (image_size,image_size,3),name = 'image_input')
outputVGG16 = vgg_16_conv(img_in)
X = Flatten()(outputVGG16)
X = Dense(2048, activation='relu',name="dense1")(X)
X = Dense(2048, activation='relu',name="dense2")(X)
X = Dense(120, activation='softmax', name="output")(X)
adapted_VGG16 = Model(input = img_in,output = X)
adapted_VGG16.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'] )

adapted_VGG16.fit(X_train,y_train,epochs = 10, batch_size = 19)
