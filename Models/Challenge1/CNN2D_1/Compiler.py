from keras.optimizers import RMSprop


# Compile the CNN2D_1 model with it's intended compilation options
def compile_model(model):
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # Tweak the optimizer

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])  # Compile model

    return model
