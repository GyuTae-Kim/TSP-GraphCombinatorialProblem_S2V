import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data_loader import DataLoader


class DataGenerator(Sequence):

    def __init__(self, config):
        self.config = config

        self.length = config['train_config']['max_episode'] + config['test_config']['max_episode']

        print(' [Task] Produces a problem distribution')
        data_loader = DataLoader(config)
        self.city_list = data_loader.get_city_list()
        self.problems = data_loader.get_problem()
        print(' [Done] Problem distribution is ready')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.problems[idx]
