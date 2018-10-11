import numpy as np

path = 'Data/Other/A_Z_Handwritten/'


# Load a training set from the Kaggle DataSet .csv file (images of handwritten capital letters)
# Challenge link : https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
# Return.shape : (?, 784,), (?,)
def load_training_data():
    alphabet = [chr(ascii) for ascii in range(65, 91)]
    file = '{}A_Z_Handwritten.csv'.format(path)
    print('Loading raw data from: "{}"'.format(file))
    dataset = np.loadtxt(file, delimiter=',')
    print('Raw data loaded')
    print('Mapping raw data')
    x, y = dataset[:, 0:784], dataset[:, 0]
    assert len(alphabet) >= len(np.unique(y))
    y = np.array([chr(int(65+label)) for label in y])
    print("Raw data mapped")
    return x, y


# Return the available output space ['A', 'B', ... 'Z']
def get_output_categories():
    return [chr(ascii) for ascii in range(65, 91)]
