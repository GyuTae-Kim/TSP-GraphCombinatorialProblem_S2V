import numpy as np
import pandas as pd

import os


class DataLoader(object):

    def __init__(self, config):
        self.config = config

        self.path = config['data_config']['path']
        self.min_city = config['data_config']['min_city']
        self.max_city = config['data_config']['max_city']

        if not os.path.exists(self.path):
            raise FileNotFoundError('[*Err] No such file: \'{}\''.format(self.path))

        print('  [Done] Successfully found city data')
        self.df = pd.read_table(self.path,
                                header=None,
                                sep=' ')
        self.df.columns = ['id', 'x', 'y']
        self.df.sort_values(by=['id'], axis=0)
        self.df.reset_index(drop=True)
        print('  [Done] Successfully Load city data')

        self.city_count = len(self.df)
        self.city_list = self.df['id'].to_numpy()

        if config['data_config']['normalize']:
            self._preprocessing()
            print('  [Done] Preprocess city data')

        feature_x = self.df['x'].to_numpy()
        feature_y = self.df['y'].to_numpy()
        self.city_info = {'list': self.city_list,
                          'feature_x': feature_x,
                          'feature_y': feature_y}

        print('  [Task] Generate TSP problems')
        self.data_length = config['train_config']['max_episode'] \
                           + config['test_config']['max_episode']
        self.n_city = np.random.randint(low=self.min_city,
                                        high=self.max_city + 1,
                                        size=self.data_length)
        self.problem = self._generate_city_problem()
        print('  [Done] Generate TSP problems')

    @property
    def get_problem(self):
        return self.problem

    @property
    def get_city_info(self):
        return self.city_info

    def _preprocessing(self):
        x_val = self.df['x'].to_numpy(dtype=np.float32)
        y_val = self.df['y'].to_numpy(dtype=np.float32)

        max_val = np.max((x_val, y_val))

        x_val = x_val / (max_val / 2.) - 1.
        y_val = y_val / (max_val / 2.) - 1.

        self.df['x'], self.df['y'] = x_val, y_val

    def _generate_city_problem(self):
        idx = np.arange(self.city_count)
        problems_idx = [np.sort(np.random.choice(idx, size=s, replace=False))
                       for s in self.n_city]

        return problems_idx
