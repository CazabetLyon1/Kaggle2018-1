import math
import numpy


# Flatten an array of n images to an array of n normalized vectors of pixels
# Input.shape : (n, x, y,)
# Output.shape : (n, x*y,)
def to_normalized_vector_list(images):
    if len(images.shape) == 2:
        return images/255
    elif len(images.shape) == 3:
        pixels_qt = images.shape[1] * images.shape[2]            # Compute x*y
        return images.reshape(images.shape[0], pixels_qt) / 255  # Flatten & normalize
    else:
        return  # TODO Handle unexpected shape


# Reconstitute an array of square image from an array of pixels vectors
# Input.shape : (n, x*y,)
# Output.shape : (n, x, y, 1)
def to_normalized_images(pixels_vectors):
    # Compute image size from its resolution
    i = math.sqrt(pixels_vectors.shape[1])
    if i != int(i):  # Ensure image square
        raise ValueError('pixels_vectors do not correspond to a square image')
    i = int(i)  # Cast to int for array dim

    return numpy.array([pixels_vector.reshape(i, i, 1) for pixels_vector in pixels_vectors])  # Format array


# Reconstitute an array of square image from an array of pixels vectors
# Input.shape : (n, x*y,) OR (n, x, y, 1)
# Output.shape : (n, x, y,)
def to_drawable_images(pixels_arrays):
    if not isinstance(pixels_arrays, (numpy.ndarray, numpy.generic)):  # Not np array
        pixels_arrays = numpy.array(pixels_arrays)  # Try to cast to np array

    # TODO Ensure correct shape and write generic pixel-size handling
    #  Get rid of the channel dim (4D array) or add the Y dim (array of pixel vectors)
    return numpy.array([pixels_array.reshape(28, 28) for pixels_array in pixels_arrays])
