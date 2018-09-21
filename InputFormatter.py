
# Flatten an array of n images to an array of n normalized vectors of pixels
# images shape : (n, x, y)
# output shape : (n, x*y)
def to_normalized_vector_list(images):
    pixels_qt = images.shape[1] * images.shape[2]          # Compute x*y
    return images.reshape(images.shape[0], pixels_qt)/255  # Flatten & nnormalize
