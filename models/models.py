def create_model(config):
    if config['model_params']['model'] == 'S2V':
        from .model_on_graph import ModelOnGraph
        model = ModelOnGraph(config)
    elif config['model_params']['model'] == 'Test1':
        from .test1_model_on_graph import ModelOnGraph
        model = ModelOnGraph(config)
    elif config['model_params']['model'] == 'Test2':
        from .test2_model_on_graph import ModelOnGraph
        model = ModelOnGraph(config)
    else:
        from .model_on_graph import ModelOnGraph
        model = ModelOnGraph(config)

    return model
