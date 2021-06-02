import numpy as np
import pandas as pd

import os


class DataLoader(object):

    def __init__(self, config):
        self.config = config

        self.feature_keys = config['data_params']['key']
        self.sort_value = config['data_params']['sort_value']
        self.path = config['data_params']['path']
        self.min_city = config['data_params']['min_city']
        self.max_city = config['data_params']['max_city']
        self.data_length = config['train_params']['max_episode'] +\
                           config['test_params']['max_episode']

        if not os.path.exists(self.path):
            raise FileNotFoundError('[*Err] No such file: \'{}\''.format(self.path))

        print(' [Done] Successfully found city data')
        self.df = pd.read_table(self.path,
                                header=None,
                                sep=' ')
        if len(self.df.columns) != len(self.feature_keys) + 1:
            raise ValueError('   [Err] Sort value\'s length must be {}'.format(len(self.df.columns) - 1))
        self.df.columns = self.sort_value + self.feature_keys
        self.df.sort_values(by=self.sort_value, axis=0)
        self.df.reset_index(drop=True)
        print(' [Done] Successfully Loaded city data')

        self.city_count = len(self.df)
        self.city_list = self.df['id'].to_numpy()

        if config['data_params']['normalize']:
            self._preprocessing()
            print(' [Done] Preprocess city data')

        self.feature = {}
        for k in self.feature_keys:
            self.feature[k] = np.reshape(self.df[k].to_numpy(),
                                         (-1, 1))
        
        self.city_info = {'list': self.city_list,
                          'feature': self.feature}

        print(' [Task] Generate TSP problems')
        self.n_city = np.random.randint(low=self.min_city,
                                        high=self.max_city + 1,
                                        size=self.data_length)
        self.problem = self._generate_city_problem()
        print(' [Done] Generated TSP problems')

    def get_problem(self):
        return self.problem

    def get_city_info(self):
        return self.city_info

    ##### If you want to use custom data, fix this part #####
    def _preprocessing(self):
        x_val = self.df['x'].to_numpy(dtype=np.float32)
        y_val = self.df['y'].to_numpy(dtype=np.float32)

        max_val = np.max((x_val, y_val))

        x_val = x_val / (max_val / 2.) - 1.
        y_val = y_val / (max_val / 2.) - 1.

        self.df['x'], self.df['y'] = x_val, y_val
    ##########################################################

    def _generate_city_problem(self):
        idx = np.arange(self.city_count)
        problems_idx = [np.sort(np.random.choice(idx, size=s, replace=False))
                       for s in self.n_city]

        return problems_idx
