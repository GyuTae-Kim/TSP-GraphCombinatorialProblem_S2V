import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data_loader import DataLoader

from graph.graph_instance import Instance

import ops


class DataGenerator(Sequence):

    def __init__(self, config):
        self.config = config

        self.feature_keys = config['data_config']['key']
        self.length = config['train_config']['max_episode'] + config['test_config']['max_episode']

        print(' [Task] Produces a problem distribution')
        data_loader = DataLoader(config)
        city_info = data_loader.get_city_info()
        self.city_list = city_info['list']
        self.feature = ops.concatenate_features(city_info['feature'],
                                                self.feature_keys)
        self.problems_idx = data_loader.get_problem()
        print(' [Done] Problem distribution is ready')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        problem_idx = self.problems_idx[idx]
        city_list = self.city_list[problem_idx]
        feature = self.feature[problem_idx]

        G = Instance(self.config, city_list, feature)

        return G, city_list
