import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import re
import pickle

import time
from PIL import Image
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
import matplotlib.pyplot as plt
import os
import itertools
import cv2
from random import shuffle


max_workers = 8

path = 'Data/Challenge2'


def gen_output_categories():
    file_names = os.listdir('{}/'.format(path))
    for file_name in file_names:
        os.rename(os.path.join('{}/'.format(path), file_name), os.path.join('{}/'.format(path),
                                                                            file_name.replace(' ', '_')))
        file_names = filter(lambda file: True if file.endswith(".csv") else False, os.listdir(path))
    file_names = [os.path.splitext(file_full_name)[0] for file_full_name in file_names]
    print(file_names)

categories = list(np.load('{}/categories.npy'.format(path)))

cat_qt = len(categories)

assert cat_qt == 340


def gen_files_size_map():
    print('Counting the lines of all files. This can take some time but only have to be done once')
    file_names = os.listdir('{}/'.format(path))
    for file_name in file_names:
        os.rename(os.path.join('{}/'.format(path), file_name), os.path.join('{}/'.format(path),
                                                                            file_name.replace(' ', '_')))
        file_names = filter(lambda file: True if file.endswith(".csv") else False, os.listdir(path))
    file_names = [os.path.splitext(file_full_name)[0] for file_full_name in file_names]

    assert len(file_names) == len(categories)  # Make sure all the the files are here

    files_size_map = {}
    for file_name in file_names:
        print('Counting rows of file: {0}/{1}.csv'.format(path, file_name))
        files_size_map[file_name] = len(pd.read_csv('{0}/{1}.csv'.format(path, file_name)))
        pass

    with open('{0}/files_size.pkl'.format(path), 'wb') as f:
        pickle.dump(files_size_map, f, pickle.HIGHEST_PROTOCOL)

    print('Counted lines of all files')


# Dictionary of the size of each source data file
files_size = None

if os.path.isfile('{}/files_size.pkl'.format(path)):         # If files_size already counted
    with open('{}/files_size.pkl'.format(path), 'rb') as f:   # Load it
        files_size = pickle.load(f)
else:                                                        # Else
    gen_files_size_map()                                     # Count length of all files


# List of all possibles country codes
country_codes = list(np.load('{}/country_codes.npy'.format(path)))

SUBSET_QT = 500
save_subsets = True

subsets_size = [None for _ in range(SUBSET_QT)]
if os.path.isfile('{}/subsets_size.pkl'.format(path)):
    with open('{}/subsets_size.pkl'.format(path), 'rb') as f:
        subsets_size = pickle.load(f)


# The resolution & ratio of the referential of the source dataset
source_resolution = {'x': 256, 'y': 256}
source_ratio = source_resolution['x'] / source_resolution['y']

# The resolution & ratio of the desired output image
target_resolution = {'x': 64, 'y': 64}
target_ratio = target_resolution['x'] / target_resolution['y']

# Referential change flag
same_resolution = source_resolution['x'] == target_resolution['x'] and source_resolution['y'] == target_resolution['y']
if not same_resolution:
    # Compute the referential projection factors (source -> target)
    x_factor = target_resolution['x'] / source_resolution['x']
    y_factor = target_resolution['y'] / source_resolution['y']


# Container for all the data relative to a drawing
class DrawingData(object):
    wrong_ratio_alerted = False

    @classmethod
    def deserialize_img_data(cls, image_data):
        # split the raw string into a list of cords (pen path)(string)
        cords = re.findall(r"\[[^\[\]]+\]", image_data)

        # Split each cord into its points
        sets = [np.int_(cord.strip('[ ]').split(',')) for cord in cords]

        # Reshape data into : (int8)[[[x1, y1], [x2, y2], ], [[x1, y1], ], ] (=== [cord:[point:[x, y], ], ])
        cords = [np.array(list(zip(sets.pop(), sets.pop())), np.uint8) for _ in range(int(math.floor(len(sets) / 2)))]
        return cords

    @classmethod
    def map_cord_points_to_lines(cls, x, y):
        return [(x[i], y[i], x[i+1], y[i+1]) for i in range(len(x)-1)]

    def __init__(self, set_id, country_code, image_data, category = None):
        set_id_type_conv_dict = {str: lambda x: np.int64(x), np.int64: lambda x: x}
        image_data_type_conv_dict = {str: lambda x: DrawingData.deserialize_img_data(x), list: lambda x: x}
        country_code_type_conv_dict = {str: lambda x: country_codes.index(x) if x != 'OTHER' else -1, int: lambda x: x,
                                       float: lambda x: -1}  # nan for some country_codes ??!! TODO: Investigate
        category_type_conv_dict = {np.str_: lambda x: categories.index(x), int: lambda x: x}
        self.set_id = set_id_type_conv_dict[type(set_id)](set_id)
        self.country_code = country_code_type_conv_dict[type(country_code)](country_code)
        self.image_data = image_data_type_conv_dict[type(image_data)](image_data)
        self.category = category_type_conv_dict[type(category)](category) if category is not None else None

    def to_image(self):
        assert target_resolution['x'] > 0 and target_resolution['y'] > 0  # Ensure minimum (1, 1) resolution

        # Function to project 2D point from source referential to target referential (scale essentially)
        projector = lambda x, y: (x, y)

        # Create the conversion functions to change referential (source to target)
        if not same_resolution:
            # Alert once if we need do change the image ratio (compress/stretch)
            if not DrawingData.wrong_ratio_alerted and target_ratio != source_ratio:
                DrawingData.wrong_ratio_alerted = True
                print('Warning, image conversion from ({0} : {1}) to ({2} : {3}).'
                      .format(source_resolution['x'], source_resolution['y'],
                              target_resolution['x'], target_resolution['y']))
                print('    Image will be compressed/stretched to comply with parameters.')

            # Functions to project 1D point from source referential to target referential for each axis (scale)
            x_projector = lambda x: x
            y_projector = lambda y: y

            if x_factor != 1:
                x_projector = lambda x: int(round(x * x_factor))
            if y_factor != 1:
                y_projector = lambda y: int(round(y * y_factor))

            projector = lambda x, y: (x_projector(x), y_projector(y))

        # Function to convert a cord to list of lines (pair of points : int[4])
        points_to_lines = lambda pts: [[pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]] for i in range(len(pts)-1)]

        # Chain the lines corresponding to each cord [(x1, y1, x2, y2), ... (xn, ...), (x'1, ...), ..., (x'n, ...), ...]
        #     Where [xi, yi] is the point n°i of the cord n°1 and [x'j, y'j] is the point n°j of the cord n°2
        lines = list(itertools.chain.from_iterable(map(points_to_lines, self.image_data)))

        # Create a np image of target dimensions
        image = np.zeros((target_resolution['x'], target_resolution['y'],), np.uint8)

        for [x1, y1, x2, y2] in lines:
            # Draw line on image with OpenCv
            cv2.line(image, projector(x1, y1)[::-1], projector(x2, y2)[::-1], 255, 1)

        return image


# Get the BEGIN and END iterator for a given subset from a given file-size
# subset_index : Index of the subset you want  0 <= {} <  SUBSET_QT
# set_size     : Size of the full set (file/category)
# return       : (uint)begin, (uint)end
def get_subset_iterators(subset_index, set_size):
    assert 0 <= subset_index < SUBSET_QT  # Ensure valid index

    subset_size = int(math.floor(float(set_size) / SUBSET_QT))  # Size of one subset
    remainder = subset_size % SUBSET_QT                         # Images to dispatch to first subsets

    # If we have to dispatch image for current subset
    remainder_to_add = 1 if subset_index < remainder else 0
    # The remainders added in previous subsets
    remainder_added = subset_index if subset_index <= remainder else remainder

    # The current subset begin iterator (uint)
    begin = subset_index * subset_size + remainder_added
    # The current subset end iterator (uint)
    end = begin + subset_size + remainder_to_add

    return begin, end


def load_subset_from_source_file(file_name, subset_index=0):
    print('Loading subset part from file: {0}/{1}.csv'.format(path, file_name))

    nb_rows = files_size[file_name] - 1   # The amount of rows in the set (-header)

    subset_begin, subset_end = get_subset_iterators(subset_index, nb_rows)  # The begin and end iterators of the subset

    data = pd.read_csv('{0}/{1}.csv'.format(path, file_name),  # Load from file
                       skiprows=(1, subset_begin+1),           # Skip from (after) header to beginning of subset
                       nrows=subset_end-subset_begin+1)        # Skip from (before) the end of the subset to EOF

    # Load raw_data and exclude header
    drawings = data["drawing"].values[1:]
    drawing_ids = list(data["key_id"].values[1:])
    drawing_countries = data["countrycode"].values[1:]

    assert len(drawings) == len(drawing_ids) == len(drawing_countries)  # Make sure all lists are the same size
    drawings_qt = len(drawings)

    # Convert raw file data as a list of DrawingData obj
    return [DrawingData(drawing_ids[i], drawing_countries[i], drawings[i], file_name)
            for i in range(drawings_qt)]

def get_subsets_size(subsets_indices=range(SUBSET_QT)):
    non_counted_indices = [i for i in subsets_indices if subsets_size[i] is None]

    for i in non_counted_indices:
        print('counting subset {}'.format(i))
        subsets_size[i] = len(load_subset(i))

    if len(non_counted_indices) != 0:
        with open('{}/subsets_size.pkl'.format(path), 'wb') as f:
            pickle.dump(subsets_size, f, pickle.HIGHEST_PROTOCOL)

    return sum([subsets_size[i] for i in subsets_indices])


def load_subset(subset_index):    # Time evaluation
    start = time.time()

    # Ensure valid subset index
    assert subset_index < SUBSET_QT

    # Check if subset previously created
    subset_path = '{0}/Subsets/{1}-{2}.pkl'.format(path, SUBSET_QT, subset_index)

    subset = None

    # Load the raw subset from files (numpy binary, or original dataset)
    if os.path.isfile(subset_path):
        #print('loading subset from file {}'.format(subset_path))
        with open(subset_path, 'rb') as f:
            subset = pickle.load(f)
    else:
        file_names = categories
        for file_name in file_names:
            if not os.path.isfile('{0}/sources/{1}.csv'.format(path, file_name)):
                print('Error, can\'t find source_file: {0}/sources/{1}.csv'.format(path, file_name))
                assert False

        assert len(file_names) >= cat_qt
        # Filter only the amount of files needed (all if cat_qt is len(categories))
        file_names = file_names if len(file_names) == cat_qt else file_names[:cat_qt]

        nested_subset = [load_subset_from_source_file(file_name, subset_index) for file_name in file_names]

        # Flatten the workers result
        subset = list(itertools.chain.from_iterable(nested_subset))

        # Shuffle the classes
        shuffle(subset)

        if save_subsets:
            with open(subset_path, 'wb') as f:
                print('saving as {}'.format(subset_path))
                pickle.dump(subset, f, pickle.HIGHEST_PROTOCOL)

    # Time evaluation
    load_time = time.time()
    load_time -= start

    # Log quantity of data loaded and time evaluation
    #print('Loaded {0} images data in: {1}s'.format(len(subset), load_time))
    return subset

def to_training_data(drawing_data_list):
    with Pool(max_workers) as p:  # Worker pool of size max_workers

        # Map data through worker pool
        x = p.map(DrawingData.to_image, drawing_data_list)
        # Map category: int to label: str (without pool)
        y = [categories[image_data.category] for image_data in drawing_data_list]

        del drawing_data_list[:]
        del drawing_data_list
        return x, y

# Load a training set from the Kaggle challenge directory containing .csv files
# Challenge link : TODO put link
# Return.shape : (?, 255, 255), (?,)
def load_training_data(subset_index=0):
    subset = load_subset(subset_index)
    return to_training_data(subset)


# Load the examination data (without labels)
# Return.shape : (?, 255, 255)
def load_prediction_data():
    file_name = 'test_simplified'
    print('Loading test dataset from file: {0}/sources/{1}.csv'.format(path, file_name))


    data = pd.read_csv('{0}/sources/{1}.csv'.format(path, file_name))

    # Load raw_data and exclude header
    drawings = data["drawing"].values
    drawing_ids = list(data["key_id"].values)
    drawing_countries = data["countrycode"].values

    assert len(drawings) == len(drawing_ids) == len(drawing_countries)  # Make sure all lists are the same size
    drawings_qt = len(drawings)

    # Convert raw file data as a list of DrawingData obj
    drawing_data_list = [DrawingData(drawing_ids[i], drawing_countries[i], drawings[i])
                         for i in range(drawings_qt)]

    with Pool(max_workers) as p:  # Worker pool of size max_workers

        # Map data through worker pool
        x = p.map(DrawingData.to_image, drawing_data_list)
        ids = [drawing_data.set_id for drawing_data in drawing_data_list]
        return ids, x


# Return the available output space ['airplane', 'alarm_clock', ... 'zigzag']
def get_output_categories():
    return categories

# Export results as specified by the challenge rules
def save_submission(results, file_name='Submission_new'):
    submission = pd.DataFrame(results)
    submission.to_csv('{0}/{1}.csv'.format(path, file_name), index=False)                           # Save submission


if __name__ == '__main__':
    # Tests, not meant to be used directly from shell

    #data = load_subset(0)
    #print(categories[data[3501].category])
    #plt.imshow(data[3501].to_image(), cmap='gray')
    for i in range(1, 50):
        data_x, data_y = load_training_data(0)
    #plt.imshow(data_x[0], cmap='gray')
    #plt.show()
