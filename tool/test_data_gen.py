import numpy as np

import argparse
import yaml
import os

from data_loader import DataLoader
from data_generator import DataGenerator


def compute_config(config, args):
    keys = config['data_params']['key']
    config['model_params']['p'] = len(keys)

    if args.one_per_one:
        config['train_params']['max_episode'] = config['data_params']['max_city'] -\
                                                config['data_params']['min_city'] + 1
    else:
        config['train_params']['max_episode'] = args.n
    config['test_params']['max_episode'] = 0


def make_one_per_one_problem(data_loader, config):
    min_city = config['data_params']['min_city']
    max_city = config['data_params']['max_city']
    n_city = np.arange(min_city, max_city + 1)
    city_count = data_loader.city_count

    idx = np.arange(city_count)
    data_loader.problem = [np.sort(np.random.choice(idx, size=s, replace=False))
                           for s in n_city]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--one_per_one', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    compute_config(config, args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    data_loader = DataLoader(config)

    if args.one_per_one:
        make_one_per_one_problem(data_loader, config)

    data_gen = DataGenerator(config, data_loader)

    for G in data_gen:
        info = G.instance_info()
        node_list = info['node_list']
        feature = info['feature']
        path = os.path.join(args.save_path, '{}_tsp.txt'.format(len(node_list)))

        with open(path, 'w') as f:
            for node, feature in zip(node_list, feature):
                data = '{} {} {}\n'.format(node, feature[0], feature[1])
                f.write(data)
