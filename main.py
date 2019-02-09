import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator


# VGG 16 model built with parameters according to the images
def pretrain_VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(60, 60, 3), name = 'zpd'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name = True)

    return model

# VGG 19 model built with parameters according to our input images
def pretrain_VGG_19():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(60, 60, 3), name = 'zpd'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.load_weights('../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name = True)

    return model

# Basic CNN structure built usign the parameters mentioned in the method
def basic_CNN():
    filters = 8 # 8 convolution filters used in a layer
    # Pooling size taken as 2*2 grid
    pool_size = 2
    # Kernel size for convolution 4*4
    conv_size = 4
    model = Sequential()
    model.add(Convolution2D(filters, conv_size, conv_size,
                            border_mode='valid',
                            input_shape=(60, 60, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(filters, conv_size, conv_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=0, momentum=0.45, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,  metrics=['accuracy'])
    return model    
    

# This method adds fully connected dense layers and a softmax layer to output class prediction on top
#  of VGG16 and VGG19 models
def transfer_learn(model):
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model
    
# Processing train data by one hot encoding and normalizing
def data_preprocess(train_X, validation_X, train_y, validation_y):
    
    train_X = np.array(train_X, dtype=np.uint8)
    train_y = np.array(train_y, dtype=np.uint8)
    
    validation_X = np.array(validation_X, dtype=np.uint8)
    validation_y = np.array(validation_y, dtype=np.uint8)
    
    train_X = train_X.transpose((0, 1, 2, 3))
    validation_X = validation_X.transpose((0, 1, 2, 3))
    
    train_X = train_X.astype('float16')
    validation_X = validation_X.astype('float16')
    
    mean_pixel = [103.939, 116.779, 123.68]
    train_X[:, 0, :, :] -= mean_pixel[0]
    train_X[:, 1, :, :] -= mean_pixel[1]
    train_X[:, 2, :, :] -= mean_pixel[2]
    
    validation_X[:, 0, :, :] -= mean_pixel[0]
    validation_X[:, 1, :, :] -= mean_pixel[1]
    validation_X[:, 2, :, :] -= mean_pixel[2]
    
    train_y = np_utils.to_categorical(train_y, 10)
    validation_y = np_utils.to_categorical(validation_y, 10)
    
    return train_X, validation_X, train_y, validation_y

# Processing test data by normalizing and reshaping
def data_preprocess_test(test_X):
    test_X = np.array(train_X, dtype=np.uint8)
    
    test_X = test_X.transpose((0, 1, 2, 3))
    
    test_X = test_X.astype('float16')
    
    mean_pixel = [103.939, 116.779, 123.68]
    test_X[:, 0, :, :] -= mean_pixel[0]
    test_X[:, 1, :, :] -= mean_pixel[1]
    test_X[:, 2, :, :] -= mean_pixel[2]
    
    return test_X


# ------------------------------------- Initializing values -----------------------------------------------
img_width, img_height = 60, 60
batch_size = 32
epochs = 10
nfolds = 10

# ----------------------------------------- Reading Driver Data --------------------------------------------

path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'driver_imgs_list.csv')
driver_imgs_list = pd.read_csv(path)
driver_imgs_list.head()
driverInImage = {}
drivers_list = []
driver_imgs = {}
classOfImage = {}
for index, row in driver_imgs_list.iterrows():
    driverInImage[row['img'].split(".")[0]] = row['subject']
    classOfImage[row['img'].split(".")[0]] = row['classname']
    if not row['subject'] in  driver_imgs:
        driver_imgs[row['subject']] = []
    driver_imgs[row['subject']].append(row['img'].split(".")[0])
    if not row['subject'] in drivers_list:
        drivers_list.append(row['subject'])

# --------------------------------------- Generate train, validation sets with unique drivers set -------------------------------
foldNum = 1
kfolds_saved_data = []
for train_drivers, validation_drivers in KFold(n_splits=nfolds, shuffle=False, random_state=None).split(drivers_list):
    validation_X = []
    validation_y = []
    train_X = []
    train_y = []
    for i in range(len(train_drivers)):
        for j in range(len(driver_imgs[drivers_list[train_drivers[i]]])):
            path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'train', classOfImage[driver_imgs[drivers_list[train_drivers[i]]][j]], driver_imgs[drivers_list[train_drivers[i]]][j] + '.jpg')
            resized_img = cv2.resize(cv2.imread(path), (img_width, img_height))
            train_X.append(resized_img)
            train_y.append(classOfImage[driver_imgs[drivers_list[train_drivers[i]]][j]][1])
    
    for i in range(len(validation_drivers)):
        for j in range(len(driver_imgs[drivers_list[validation_drivers[i]]])):
            path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'train', classOfImage[driver_imgs[drivers_list[validation_drivers[i]]][j]], driver_imgs[drivers_list[validation_drivers[i]]][j] + '.jpg')
            resized_img = cv2.resize(cv2.imread(path), (img_width, img_height))
            validation_X.append(resized_img)
            validation_y.append(classOfImage[driver_imgs[drivers_list[validation_drivers[i]]][j]][1])
    
    
    

    train_X, validation_X, train_y, validation_y = data_preprocess(train_X, validation_X, train_y, validation_y)
    
                
    print('Train shape:', train_X.shape)
    print(train_X.shape[0], 'train samples')
    
    model = pretrain_VGG_16()
    model = transfer_learn(model)
    
    validation_weights = os.path.join('validation_weights_vgg16_' + str(foldNum) + '.h5')
    
    train_datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        width_shift_range=0.2,
        zoom_range=0.2
        # height_shift_range=0.2,
        # horizontal_flip=True
        )
        
    valid_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        zoom_range=0.2
    )

    # define the grid search parameters
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
    # learning_rate = [0.001, 0.0001, 0.00001, 0.000001]
    # param_grid = dict(optimizer=optimizer, learning_rate= learning_rate)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    # grid_result = grid.fit(X, Y)


    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(train_X)
    train_datagen.fit(validation_X)
    train_generator = train_datagen.flow(train_X, train_y, batch_size=32)
    valid_generator = valid_datagen.flow(validation_X, validation_y, batch_size=32)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ModelCheckpoint(validation_weights, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    model.fit_generator(train_generator, steps_per_epoch=len(train_X) , shuffle=True, verbose=1, validation_data=valid_generator,validation_steps=len(validation_X), callbacks=callbacks)

    prediction_valid = model.predict_generator(valid_generator,steps=len(validation_X), verbose=1)
    score = log_loss(validation_y, prediction_valid)
    print('Log Loss after fold number ',foldNum, score)
    foldNum += 1
    kfolds_saved_data.append((prediction_valid, score))

avg_val_logloss = 0
for i in range(len(kfolds_saved_data)):
    avg_val_logloss += kfolds_saved_data[i][1]

print("Average K-Fold Validation Log Loss --> ", avg_val_logloss)

# ----------------------------- Read and store test data ---------------------------------------------
test_X = []
test_path = os.path.join('..', 'input/state-farm-distracted-driver-detection', 'test', '*.jpg')
test_files = glob.glob(test_path)
for file in test_files:
    resized_img = cv2.resize(cv2.imread(file), (img_width, img_height))
    test_X.append(resized_img)



#-------------------------------- Testing using the saved weights ------------------------------------
        
for i in range(test_X):
    kfold_predictions = []
    kfold_mean_prediction = []
    for j in range(nfolds):
        
        model = pretrain_VGG_16()
        
        # Load saved weights of each validation fold from the cache
        model.load_weights(os.path.join('validation_weights_vgg16' + str(j + 1) + '.h5'))
        
        # Add few layers on top and compile it
        model.transfer_learn()
        
        prediction = model.predict(test_X[i], batch_size=batch_size, verbose=1)
        kfold_predictions.append(prediction)
        
    kfold_mean_prediction.append(np.mean(kfold_predictions, axis = 0))

# Saving the preditions into a csv file by constructing dataframe with image names as indices and class names as columns
result = pd.DataFrame(kfold_mean_prediction, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'], index = names)
result.index.name = 'img'
output_file = os.path.join('submission.csv')
result.to_csv(output_file)
