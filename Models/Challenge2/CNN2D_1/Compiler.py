from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy


# Compile the CNN2D_1 model with it's intended compilation options
def compile_model(model):
    model.compile(optimizer=Adam(lr=0.0024), loss='categorical_crossentropy',
                  metrics=[categorical_crossentropy, categorical_accuracy])

    return model
