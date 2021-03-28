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
        city_info = data_loader.get_city_info()
        self.city_list = city_info['list']
        self.feature = {'x': city_info['feature_x'],
                        'y': city_info['feature_y']}
        self.problems_idx = data_loader.get_problem()
        print(' [Done] Problem distribution is ready')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        problem_idx = self.problems_idx[idx]
        city_list = self.city_list[problem_idx]
        feature_x = self.feature['x'][problem_idx]
        feature_y = self.feature['y'][problem_idx]

        return city_list, feature_x, feature_y
