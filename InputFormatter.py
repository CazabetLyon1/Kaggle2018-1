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
def to_normalized_images(images):
    if len(images.shape) == 2:
        # Compute image size from its resolution
        i = math.sqrt(images.shape[1])
        if i != int(i):  # Ensure image square
            raise ValueError('pixels_vectors do not correspond to a square image')
        i = int(i)  # Cast to int for array dim
        return numpy.array([pixels_vector.reshape(i, i, 1) for pixels_vector in images])  # Format array
    elif len(images.shape) == 3:
        return numpy.array([numpy.array(image.reshape(images.shape[1], images.shape[2], 1), numpy.float32)/255
                            for image in images])
    else:
        raise ValueError('unexpected image list shape {}'.format(images.shape))


# Reconstitute an array of square image from an array of pixels vectors
# Input.shape : (n, x*y,) OR (n, x, y, 1)
# Output.shape : (n, x, y,)
def to_drawable_images(images):
    if not isinstance(images, (numpy.ndarray, numpy.generic)):  # Not np array
        images = numpy.array(images)  # Try to cast to np array
    if len(images.shape) == 2:
        # TODO Ensure correct shape and write generic pixel-size handling
        #  Get rid of the channel dim (4D array) or add the Y dim (array of pixel vectors)
        return numpy.array([pixels_array.reshape(28, 28) for pixels_array in images])
    elif len(images.shape) == 3:
        return numpy.array([numpy.array(image.reshape(images.shape[1], images.shape[2], ), numpy.float32) / 255
                            for image in images])
    elif len(images.shape) == 4:
        return numpy.array(numpy.round(images.reshape(images.shape[0], images.shape[1], images.shape[2], )*255), numpy.float32)
    else:
        raise ValueError('unexpected image list shape {}'.format(images.shape))
