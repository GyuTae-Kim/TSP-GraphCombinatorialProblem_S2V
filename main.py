import argparse
import yaml

from data_loader import DataLoader
from data_generator import DataGenerator
from graph_handler import GraphHandler
from models.model_on_graph import ModelOnGraph
from agent import Agent


def compute_config(config, args):
    keys = config['data_params']['key']
    config['model_params']['p'] = len(keys)

    if args.test_only:
        config['train_params']['max_episode'] = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    compute_config(config, args)

    print('[Task] Load Data Loader')
    data_loader = DataLoader(config)
    print('[Done] Loaded Data Loader')
    print('[Task] Load Data Generator')
    data_gen = DataGenerator(config, data_loader)
    print('[Done] Loaded Data Generator')
    print('[Task] Load Graph Handler')
    graph_handler = GraphHandler(config, data_gen)
    print('[Done] Loaded Graph Handler')
    print('[Task] Build Model')
    model_on_graph = ModelOnGraph(config)
    print('[Done] Built Model')
    print('[Task] Load Agent')
    agent = Agent(config, graph_handler, model_on_graph)
    print('[Done] Loaded Agent')

    if args.test_only:
        agent.run_test()
    else:
        agent.running()
