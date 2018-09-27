

# Compile the MLP1D_2 model with it's intended compilation options
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model

    return model
