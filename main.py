import numpy as np

import argparse
import yaml
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_loader import DataLoader
from data_generator import DataGenerator
from graph_handler import GraphHandler
from models.models import create_model
from agent import Agent
from utils.memory import Memory as Mem


def compute_config(config, args):
    keys = config['data_params']['key']
    config['model_params']['p'] = len(keys)
    config['test_params']['save_test_log'] = False
    config['test_params']['test_path'] = None

    if not os.path.exists('results'):
        os.mkdir('results')
    
    if args.test_only:
        config['train_params']['max_episode'] = 0
        if args.test_data_path is not None:
            config['test_params']['save_test_log'] = True
            config['test_params']['test_result_path'] = 'results/test_result.txt'

def data_genenrator_from_data(data_gen, args, config):
    problems_idx, feature = read_files(args.test_data_path)
    data_gen.problems_idx = problems_idx
    data_gen.feature = feature
    if args.test_only:
        config['test_params']['max_episode'] = len(problems_idx)

def read_files(path):
    find = os.path.join(path, '*_tsp.txt')
    f_list = glob.glob(find)

    problems_idx = []
    feature = []
    last_idx = 0

    for f_path in f_list:
        with open(f_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split(' ')
                feature.append([l[1], l[2]])
            problems_idx.append(np.arange(last_idx, len(lines) + last_idx))
            last_idx += len(lines) - 1
    feature = np.array(feature, dtype=np.float32)
    
    return problems_idx, feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_data_path', type=str, default=None)
    args = parser.parse_args()
    if args.test_data_path is not None:
        if not os.path.exists(args.test_data_path):
            raise FileNotFoundError('[*Err] No such file: \'{}\''.format(args.test_data_path))
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    compute_config(config, args)

    print('[Task] Load Data Loader')
    data_loader = DataLoader(config)
    print('[Done] Loaded Data Loader')
    print('[Task] Load Data Generator')
    data_gen = DataGenerator(config, data_loader)
    if args.test_only and args.test_data_path is not None:
        data_genenrator_from_data(data_gen, args, config)
    print('[Done] Loaded Data Generator')
    print('[Task] Load Graph Handler')
    mem = Mem(config, data_gen)
    graph_handler = GraphHandler(config, data_gen, mem)
    print('[Done] Loaded Graph Handler')
    print('[Task] Build Model')
    model_on_graph = create_model(config)
    print('[Done] Built Model')
    print('[Task] Load Agent')
    agent = Agent(config, graph_handler, model_on_graph)
    if not args.test_only and config['train_params']['test_while_training']:
        test_data_gen = DataGenerator(config, data_loader)
        data_genenrator_from_data(test_data_gen, args, config)
        agent.set_test_data_gen(test_data_gen)
    print('[Done] Loaded Agent')

    if args.test_only:
        agent.run_test()
    else:
        agent.running()
