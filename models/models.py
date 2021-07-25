def create_model(config):
    if config['model_params']['model'] == 'S2V':
        from .s2v_dqn import S2V_DQN
        model = S2V_DQN(config)
    elif config['model_params']['model'] == 'Test1':
        from .test1_model_on_graph import S2V_DQN
        model = S2V_DQN(config)
    else:
        from .s2v_dqn import S2V_DQN
        model = S2V_DQN(config)

    return model
