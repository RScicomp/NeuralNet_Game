import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
from random import randint

import matplotlib.pyplot as plt

from game import controlled_run

# importing static variables
from game import DO_NOTHING
from game import JUMP

