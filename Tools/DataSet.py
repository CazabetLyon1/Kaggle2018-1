from keras.datasets import mnist
import random
import numpy as np
import pandas
import math


# Object containing all the inputs and expected outputs both for training and testing
class DataSet(object):
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    # Split a given set of images & labels into a training set and a testing set
    @classmethod
    def split_set(cls, images, labels, vp=0.3):
        # Handle type errors / implicit cast
        if not isinstance(images, (list,)):
            if isinstance(images, (np.ndarray, np.generic)):
                images = list(images)
            else:
                return  # TODO Handle type error
        if not isinstance(labels, (list,)):
            if isinstance(labels, (np.ndarray, np.generic)):
                labels = list(labels)
            else:
                return  # TODO Handle type error
        validation_qt = int(math.ceil(len(images)*vp))  # Compute the amount of data reserved for validation

        sampled_indexes = random.sample(range(len(images)), validation_qt)  # validation_qt of unique idx of data

        # Copy data corresponding to the random indexes into the test set
        test_images = [images[i] for i in sampled_indexes]
        test_labels = [labels[i] for i in sampled_indexes]

        sampled_indexes.sort(key=int, reverse=True)  # Sort indices desc to avoid deleting the wrong element

        # Remove the images from the training set
        for i in sampled_indexes:
            images.pop(i)
            labels.pop(i)

        return np.array(images), np.array(labels), np.array(test_images), np.array(test_labels)  # Return the set tuple

    @classmethod
    def load_from_mnist(cls):
        # load the DataSet
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return DataSet(train_images, train_labels, test_images, test_labels)

    # Load a training set from the Kaggle challenge .csv file
    # @param vp : The proportion of the set to use for training validation
    # Data_Shape : (n*(1/vp), pix_qt,), (n*(1/vp),), (n*vp, pix_qt,), (n*vp,)
    @classmethod
    def load_from_csv(cls, path, vp=0.3):
        full_set = pandas.read_csv('{0}/train.csv'.format(path))  # Load the csv
        labels = np.array(full_set["label"])                      # Extract all labels
        images = np.array(full_set.drop(['label'], axis=1))       # Extract all images
        del full_set

        # Split the set into Training & Testing sets
        train_images, train_labels, test_images, test_labels = cls.split_set(images, labels, vp)

        return DataSet(train_images, train_labels, test_images, test_labels)  # Instantiate the DataSet

    # Encode the DataSet using the functions given as parameter and return it as new DataSet
    def encode(self, images_encoding_fct=None, labels_encoding_fct=None):
        # Apply the image encoding if encoding function in parameters. If not, pass data through
        if images_encoding_fct is not None:
            train_images = images_encoding_fct(self.train_images)
            test_images = images_encoding_fct(self.test_images)
        else:
            train_images = self.train_images
            test_images = self.test_images

        # Apply the label encoding if encoding function in parameters. If not, pass data through
        if labels_encoding_fct is not None:
            train_labels = labels_encoding_fct(self.train_labels)
            test_labels = labels_encoding_fct(self.test_labels)
        else:
            train_labels = self.train_labels
            test_labels = self.test_labels

        # Return the new DataSet with encoded (or not) values
        return DataSet(train_images, train_labels, test_images, test_labels)
