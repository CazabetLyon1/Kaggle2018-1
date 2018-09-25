import math
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.optimizers import RMSprop


# Multilayer perceptron n°1
# Input shape  : (pixels_qt,)
# Output shape : (classes_qt,)
def mlp1D1_initializer(pixels_qt, classes_qt):
    model = Sequential()  # Instantiate model

    model.add(Dense(pixels_qt, input_dim=pixels_qt, kernel_initializer='normal', activation='relu'))  # Input layer
    model.add(Dense(classes_qt, kernel_initializer='normal', activation='softmax'))  # Output layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model

    return model


# Multilayer perceptron n°2
# Input shape  : (pixels_qt,)
# Output shape : (classes_qt,)
def mlp1D2_initializer(pixels_qt, classes_qt):
    model = Sequential()  # Instantiate model

    # Input logic layer
    model.add(Dense(pixels_qt, input_dim=pixels_qt, kernel_initializer='normal', activation='relu'))
    # Inner logic layer
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    # Inner logic layer
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    # Output logic layer
    model.add(Dense(classes_qt, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model

    return model


# Convolutional n°2
# Input shape  : (sqrt(pixels_qt), sqrt(pixels_qt),)
# Output shape : (classes_qt,)
def cnn2D1_initializer(pixels_qt, classes_qt):
    print(pixels_qt)
    i = math.sqrt(pixels_qt)
    if i != int(i):
        raise ValueError('pixels_vectors do not correspond to a square image')
    i = int(i)
    print(i)
    model = Sequential()  # Instantiate model

    # Convolution input layer
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(i, i, 1)))
    # Convolution layer
    model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='Same', activation='relu'))
    # Layer pooling from a 2² kernel on maximum
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Remove (Drop) 25% of neurons regularly (prevent overfitting)
    model.add(Dropout(0.25))

    # Convolution layer
    model.add(Conv2D(filters=56, kernel_size=(4, 4), padding='Same', activation='relu'))
    # Convolution layer
    model.add(Conv2D(filters=56, kernel_size=(2, 2), padding='Same', activation='relu'))
    # Remove (Drop) 25% of neurons regularly (prevent overfitting)
    model.add(Dropout(0.25))

    model.add(Flatten())                                # Reshape neurons to 1D vector
    model.add(Dense(224, activation="relu"))            # Inner logic layer
    model.add(Dropout(0.2))                             # Remove (Drop) 20% of neurons regularly (prevent overfitting)
    model.add(Dense(75, activation="relu"))             # Inner logic layer
    model.add(Dropout(0.2))                             # Remove (Drop) 20% of neurons regularly (prevent overfitting)
    model.add(Dense(classes_qt, activation="sigmoid"))  # Output logic layer

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # Tweak the optimizer

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])  # Compile model

    return model
