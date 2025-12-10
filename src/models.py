# -*- coding: utf-8 -*-
"""
This file defines the models to be used for the classification of ECG signals
in the Brugada dataset proposed in [1].

[1] M. Scarpiniti and A. Uncini, "Exploiting phase information for the
identification of Brugada syndrome: A preliminary study", in Italian Workshop
on Neural Networks (WIRN 2024), Vietri sul Mare (SA), Italy, June 05-07, 2024.


Created on Mon Apr 15 23:20:03 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


# TensorFlow ≥2.0 is required
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Rescaling, concatenate
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import GaussianNoise, Activation
from tensorflow.keras.layers import Conv1D, LSTM



# Create the 1D CNN + DNN using L leads with N-point magnitude and phase
def DNN_CNN(LR=0.001, L=1, N=128):
    """
    Create the proposed model composed of three branches:
    - the first for the ECG time domain (cascade of 1D Conv layers)
    - the second for the FFT ECG magnitude (cascade of Dense layers)
    - the third for the FFT ECG phase (cascade of Dense layers)

    Parameters
    ----------
    LR : (float) Learning rate. The default is 0.001.
    L : (int) Number of ECG leads. The default is 1.
    N : (int) Number of FFT nibs. The default is 128.

    Returns
    -------
    net : TensorFlow Model object.
    """

    # Input definition
    in_x = Input(shape=(150,L))
    in_m = Input(shape=(N*L,))
    in_p = Input(shape=(N*L,))

    # First branch
    x_x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(in_x)
    x_x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(x_x)
    x_x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(x_x)
    x_x = Dropout(0.2)(x_x)
    x_x = LSTM(200)(x_x)

    # Second branch
    x_m = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(in_m)
    x_m = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x_m)
    x_m = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x_m)
    x_m = GaussianNoise(0.1)(x_m)
    x_m = Activation('relu')(x_m)
    x_m = BatchNormalization()(x_m)

    # Third branch
    x_p = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(in_p)
    x_p = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x_p)
    x_p = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x_p)
    x_p = GaussianNoise(0.1)(x_p)
    x_p = Activation('relu')(x_p)
    x_p = BatchNormalization()(x_p)

    # Concatenation (INTERMEDIATE DATA FUSION)
    x = concatenate([x_x, x_m, x_p], axis=1)

    # Output layers
    y = Dense(25, activation='relu')(x)
    y = Dense(1, activation='sigmoid')(y)

    # Model definition
    net = Model([in_x, in_m, in_p], y)


    # Display the model's architecture
    # net.summary()

    net.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy'])

    return net



# Create the DNN using L leads
def DNN(LR=0.001, L=1):
    """
    Create the DNN model for comparison.

    Parameters
    ----------
    LR : (float) Learning rate. The default is 0.001.
    L : (int) Number of ECG leads. The default is 1.

    Returns
    -------
    net : TensorFlow Model object.
    """

    in_x = Input(shape=(150*L,))

    x = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(in_x)
    x = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x)
    x = Dense(5, kernel_regularizer=keras.regularizers.L1(1E-2))(x)
    x = GaussianNoise(0.1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    y = Dense(1, activation='sigmoid')(x)

    # Model definition
    dnn = Model(in_x, y)

    # Display the model's architecture
    # dnn.summary()

    dnn.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy'])

    return dnn



# Create the 1D CNN using L leads
def CNN(LR=0.001, L=1):
    """
    Create the CNN model for comparison.

    Parameters
    ----------
    LR : (float) Learning rate. The default is 0.001.
    L : (int) Number of ECG leads. The default is 1.

    Returns
    -------
    net : TensorFlow Model object.
    """

    in_x = Input(shape=(150,L))

    x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(in_x)
    x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(x)
    x = Conv1D(filters=10, kernel_size=8, strides=1, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = LSTM(200)(x)

    y = Dense(1, activation='sigmoid')(x)

    # Model definition
    cnn = Model(in_x, y)

    # Display the model's architecture
    # cnn.summary()

    cnn.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy'])

    return cnn



# Create the S3 AlexNet-based architecture with 2xL channels for early fusion
def S3(LR=0.001, L=1):
    """
    Create the S3 model for comparison.

    Parameters
    ----------
    LR : (float) Learning rate. The default is 0.001.
    L : (int) Number of ECG leads. The default is 1.

    Returns
    -------
    net : TensorFlow Model object.
    """

    in_x = Input(shape=(224,224,2*L))

    x = Rescaling(scale=1./255)(in_x)

    x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

    x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

    z = Flatten()(x)
    z = Dense(4096, activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Dense(4096, activation='relu')(z)
    z = Dropout(0.5)(z)

    y = Dense(1, activation='sigmoid')(z)

    # Model definition
    net = Model(in_x, y)

    # Display the model's architecture
    # net.summary()

    net.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net



# Create the S5 AlexNet-based architecture with intermediate data fusion before Dense layers
def S5(LR=0.001, L=1):
    """
    Create the S5 model for comparison.

    Parameters
    ----------
    LR : (float) Learning rate. The default is 0.001.
    L : (int) Number of ECG leads. The default is 1.

    Returns
    -------
    net : TensorFlow Model object.
    """

    # Input definition
    in_m = Input(shape=(224, 224, L))
    in_p = Input(shape=(224, 224, L))

    # First branch
    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    y_m = Flatten()(x_m)


    # Second branch
    x_p = Rescaling(scale=1./255)(in_p)

    x_p = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    y_p = Flatten()(x_p)


    # Concatenation (INTERMEDIATE DATA FUSION)
    y = concatenate([y_m, y_p], axis=1)


    # Classifier
    y = Dense(2048, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(256, activation='relu')(y)
    y = Dropout(0.5)(y)
    output = Dense(1, activation='sigmoid')(y)


    # Model definition
    net = Model([in_m, in_p], output)

    # Display the model's architecture
    # net.summary()

    net.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net
