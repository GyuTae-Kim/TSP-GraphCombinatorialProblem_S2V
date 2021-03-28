import numpy as np
import pandas as pd

import os


class DataLoader():

    def __init__(self, config):
        self.config = config

        self.path = config['data_config']['path']
        self.min_city = config['data_config']['min_city']
        self.max_city = config['data_config']['max_city']

        if not os.path.exists(self.path):
            raise FileNotFoundError('[*Err] No such file: \'{}\''.format(self.path))

        print('  [Done] Successfully found city data')
        self.table = self.table = pd.read_table(self.path,
                                                header=None,
                                                sep=' ')
        print('  [Done] Successfully Load city data')
        self.table.columns = ['id', 'x', 'y']
        self.city_count = len(self.table)
        self.city_list = self.table['id'].to_numpy()

        if config['data_config']['normalize']:
            self._preprocessing()
            print('  [Done] Preprocess city data')

        self.data_length = config['train_config']['max_episode'] \
                           + config['test_config']['max_episode']
        self.n_city = np.random.randint(low=self.min_city,
                                        high=self.max_city + 1,
                                        size=self.data_length)
        self.problem = self._generate_city_problem()
        print('  [Done] Generate TSP problems')

    @property
    def get_city_list(self):
        return self.city_list

    @property
    def get_problem(self):
        return self.problem

    def _preprocessing(self):
        x_val = self.table['x'].to_numpy(dtype=np.float32)
        y_val = self.table['y'].to_numpy(dtype=np.float32)

        max_val = np.max((x_val, y_val))

        x_val = x_val / (max_val / 2.) - 1.
        y_val = y_val / (max_val / 2.) - 1.

        self.table['x'], self.table['y'] = x_val, y_val

    def _generate_city_problem(self):
        problem = [np.random.choice(self.city_list, size=s, replace=False)
                   for s in self.n_city]

        return problem
