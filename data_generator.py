import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data_loader import DataLoader

from graph.graph_instance import Instance

import ops


class DataGenerator(Sequence):

    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.feature_keys = config['data_params']['key']
        self.length = config['train_params']['max_episode'] +\
                      config['test_params']['max_episode']

        print(' [Task] Produces problem distribution')
        city_info = data_loader.get_city_info()
        self.feature = ops.concatenate_features(city_info['feature'],
                                                self.feature_keys)
        self.problems_idx = data_loader.get_problem()
        print(' [Done] Problem distribution is ready')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        problem_idx = self.problems_idx[idx]
        n_city = len(problem_idx)
        feature = self.feature[problem_idx]

        G = Instance(n_city, feature)

        return G
