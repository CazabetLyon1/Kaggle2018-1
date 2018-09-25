import pandas
import numpy

path = 'Data/Challenge1/'


# Load a training set from the Kaggle challenge .csv file
# Challenge link : https://www.kaggle.com/c/digit-recognizer
# Return.shape : (42000, 784,), (42000,)
def load_training_data():
    read = pandas.read_csv('{0}train.csv'.format(path))  # Load the csv
    labels = numpy.array(read["label"])                  # Extract all labels
    images = numpy.array(read.drop(['label'], axis=1))   # Extract all images
    del read                                             # Free file

    return images, labels


# Load the examination data (without labels)
# Return.shape : (28001, 784,)
def load_prediction_data():
    read = pandas.read_csv('{0}test.csv'.format(path))  # Load the csv
    images = numpy.array(read)                          # Extract all images
    del read                                            # Free file

    return images


# Export results as specified by the challenge rules
# param.shape : (28001,)
def save_submission(results, file_name='Submission_new'):
    results = pandas.Series(results, name='Label')                                                 # Cast results
    submission = pandas.concat([pandas.Series(range(1, 28001), name='ImageId'), results], axis=1)  # Put image indices
    submission.to_csv('{0}{1}.csv'.format(path, file_name), index=False)                           # Save submission
