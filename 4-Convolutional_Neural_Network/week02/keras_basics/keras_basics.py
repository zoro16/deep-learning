import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, \
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    conv_filters = 32
    conv_kernel_size = (7, 7)
    conv_strides = (1, 1)
    conv_name = 'conv0'
    # CONV -> BN -> RELU Block applied to X
    # keras.layers.Conv2D(
    #     filters, kernel_size, strides=(1, 1), padding='valid',
    #     data_format=None, dilation_rate=(1, 1), activation=None,
    #     use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #     kernel_constraint=None, bias_constraint=None
    # )
    X = Conv2D(conv_filters, conv_kernel_size, strides=conv_strides, name=conv_name )(X)

    # keras.layers.BatchNormalization(
    #     axis=-1, momentum=0.99, epsilon=0.001, center=True,
    #     scale=True, beta_initializer='zeros', gamma_initializer='ones',
    #     moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #     beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #     gamma_constraint=None
    # )
    X = BatchNormalization(axis= 3, name='bn0')(X)

    X = Activation('relu')(X)

    # MAXPOOL
    # keras.layers.MaxPooling2D(
    #     pool_size=(2, 2), strides=None, padding='valid', data_format=None
    # )
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)

    # keras.layers.Dense(
    #     units, activation=None, use_bias=True,
    #     kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #     kernel_regularizer=None, bias_regularizer=None,
    #     activity_regularizer=None, kernel_constraint=None, bias_constraint=None
    # )
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


# call the model
happyModel = model(X_train.shape[1:])

# compile the model to configure the learning process
happyModel.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model
happyModel.fit(X_train, Y_train, epochs=40, batch_size=32)


# test/evaluate the model
# preds = happyModel.evaluate(X_test, Y_test, batch_size=32)

# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))


# # # HOW TO USE THE TRAINED MODEL
# img_path = 'images/16006860405_de11619ec9_b.jpg'
# img = image.load_img(img_path, target_size=(64, 64))
# imshow(img)

# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# print(happyModel.predict(x))
