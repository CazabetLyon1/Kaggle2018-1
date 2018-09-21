import math
import numpy


# Flatten an array of n images to an array of n normalized vectors of pixels
# Images shape : (n, x, y,)
# Output shape : (n, x*y,)
def to_normalized_vector_list(images):
    pixels_qt = images.shape[1] * images.shape[2]          # Compute x*y
    return images.reshape(images.shape[0], pixels_qt)/255  # Flatten & nnormalize


# Reconstitute an array of square image from an array of pixels vectors
# Vectors shape : (n, x*y,)
# Output  shape : (n, x, y,)
def to_normalized_images(pixels_vectors):
    i = math.sqrt(pixels_vectors.shape[1])
    if i != int(i):
        raise ValueError('pixels_vectors do not correspond to a square image')

    i = int(i)
    return numpy.array([pixels_vector.reshape(i, i) for pixels_vector in pixels_vectors])
