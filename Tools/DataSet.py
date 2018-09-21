from keras.datasets import mnist


# Object containing all the inputs and expected outputs both for training and testing
class DataSet(object):
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    @classmethod
    def load_from_mnist(cls):
        # load the DataSet
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return DataSet(train_images, train_labels, test_images, test_labels)

    # Encode the DataSet using the functions given as parameter and return it as new DataSet
    def encode(self, images_encoding_fct, labels_encoding_fct):
        return DataSet(images_encoding_fct(self.train_images), labels_encoding_fct(self.train_labels),
                       images_encoding_fct(self.test_images), labels_encoding_fct(self.test_labels))
