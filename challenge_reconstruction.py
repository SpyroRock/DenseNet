from __future__ import print_function

#import warnings

#from distutils.version import LooseVersion
#import tensorflow as tf

import numpy
from numpy import load, save
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras import backend as K

#assert LooseVersion(tf.__version__) >= LooseVersion('1.0')
#print('Tensorflow Version: {}'.format(tf.__version__))

#Check for a GPU
#if not tf.test.gpu_device_name():
#    warnings.warn('No GPU found')
#else:
#    print('Deafault GPU Device: {}'.format(tf.test.gpu_device_name()))

img_rows, img_cols = 64, 64
img_height_test, img_width_test = 64, 64

speckle_data = load('speckle_array_case0.npy')
print(speckle_data.shape)
#speckle_labels = load('speckle_labels.npy')
speckle_labels = load('symbol_array_case0.npy')
print(speckle_labels.shape)
#plt.imshow(speckle_labels[2], cmap='gray')
#plt.show()
#dictionary = {speckle_labels_n: speckle_labels_mn_n for speckle_labels_n, speckle_labels_mn_n in zip(speckle_labels, speckle_labels_mn)}

trainX, testX, Y_train, Y_test = train_test_split(speckle_data, speckle_labels, test_size=0.1, random_state=42)

trainX = trainX.reshape(-1, img_rows, img_cols, 1)
testX = testX.reshape(-1, img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Y_train = Y_train.reshape(-1, img_height_test, img_width_test, 1)
Y_test = Y_test.reshape(-1, img_height_test, img_width_test, 1)
input_shape_test = (img_height_test, img_width_test, 1)

batch_size = 100
nb_classes = 6
nb_epoch = 5

img_channels = 1

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
print(img_dim)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation

# model = densenet.DenseNetFCN(input_shape=img_dim, depth=depth, include_top=True, nb_dense_block=nb_dense_block, classes=1, nb_filter=-1, 
#                             growth_rate=16, dropout_rate=dropout_rate)

model = densenet.DenseNetFCN(input_shape=img_dim)

print("Model created")

model.summary()
optimizer = Adam(lr=1e-3) # Using Adam instead of SGD to speed up training
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

print('After preprocess_input: {}'.format(trainX.shape))
print('After preprocess_input: {}'.format(testX.shape))

model.fit(trainX, Y_train, 
          batch_size = 50, 
          epochs = 10, 
          verbose = 1, 
          validation_data = (testX, Y_test)) # Data on which to evaluate the loss and any model metrics at the end of each epoch. 
                                              # The model will not be trained on this data. 
                                              # This could be a list (x_val, y_val) or a list (x_val, y_val, val_sample_weights). 
                                              # validation_data will override validation_split.

score = model.evaluate(testX, Y_test, verbose = 0)

print('Test loss:', score[0])
print('Test acuracy:', score[1]) 

y_predicted = model.predict(testX)

print(y_predicted.shape)
save('challenge_predicted.npy', y_predicted)

# extract = Model(reconstruction.inputs, reconstruction.layers[-1].output) # Dense(128,...)
# features = extract.predict(testX)
# print(features.shape)
# save('features_data', features)
# save('features_predicted', y_predicted)

### ----------------------------------------------------------------------------------------------- ###
###                  Original Code                                                                  ###

# generator = ImageDataGenerator(rotation_range=15,
#                                width_shift_range=5./32,
#                                height_shift_range=5./32,
#                                horizontal_flip=True)

# generator.fit(trainX, seed=0)

# # Load model
# weights_file="weights/DenseNet-40-12-CIFAR10.h5"
# if os.path.exists(weights_file):
#     #model.load_weights(weights_file, by_name=True)
#     print("Model loaded.")

# out_dir="weights/"

# lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
#                                     cooldown=0, patience=5, min_lr=1e-5)
# model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
#                                   save_weights_only=True, verbose=1)

# callbacks=[lr_reducer, model_checkpoint]

# model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
#                     steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
#                     # callbacks=callbacks,
#                     validation_data=(testX, Y_test),
#                     validation_steps=testX.shape[0] // batch_size, verbose=1)

# yPreds = model.predict(testX)
# print(yPreds.shape)
# save('challenge_predicted.npy', yPreds)
# yPred = np.argmax(yPreds, axis=1)
# yTrue = testY

# accuracy = metrics.accuracy_score(yTrue, yPred) * 100
# error = 100 - accuracy
# print("Accuracy : ", accuracy)
# print("Error : ", error)

