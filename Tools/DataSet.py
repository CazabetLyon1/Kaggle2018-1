import math
import numpy as np
import random
from Tools.ThreadedGenerator import *

# Object containing all the inputs and expected outputs both for training and testing
class DataSet(object):
    def __init__(self, train_generator, val_generator):
        self.train_generator = train_generator
        self.val_generator = val_generator