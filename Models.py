from keras.models import Sequential
from keras.layers import Dense


# Multilayer perceptron n°1
# Input shape  : (pixels_qt,)
# Output shape : (classes_qt,)
def mlp1D1_initializer(pixels_qt, classes_qt):
    model = Sequential()  # Instantiate model

    model.add(Dense(pixels_qt, input_dim=pixels_qt, kernel_initializer='normal', activation='relu'))  # Input layer
    model.add(Dense(classes_qt, kernel_initializer='normal', activation='softmax'))                   # Output layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model

    return model
