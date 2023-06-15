def no_q():
    fname_grid = [
        'entro_0',
        'g0_-2_entro0',
        'g0.5_-3_entro0',
        ]
    loss_weights_grid = [
        [0, 1E0, 1E0, 0],
        [1E-2, 1E0, 1E0, 0],
        [1E-3, 1E0, 1E0, 0],
        ]
    param_updates = [
        {},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        ]
    return fname_grid, loss_weights_grid, param_updates

def test_full():
    fname_grid = [
        'mf0',
        'entro-1',
        'entro1',
        'entro0',
        'g0_-2_entro-1',
        'g0.25_-3_entro-1',
        'g0.5_-3_entro-1',
        'g0.8_-4_entro-1',
        'g0_-2_entro0',
        'g0.25_-3_entro0',
        'g0.5_-3_entro0',
        'g0.8_-4_entro0',
        ]
    loss_weights_grid = [
        [0, 0, 0, 1],
        [0, 1E-1, 1E-1, 1],
        [0, 1E1, 1E1, 1],
        [0, 1E0, 1E0, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-4, 1E-1, 1E-1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        {}, {}, {}, {},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates

def mf_grid():
    fname_grid = ['mf0', 'mf-1', 'mf1', 'mf2']
    loss_weights_grid = [
        [0,0,0,1E0],
        [0,0,0,1E-1],
        [0,0,0,1E1],
        [0,0,0,1E2],
        ]
    param_updates = [{}, {}, {}, {}]
    return fname_grid, loss_weights_grid, param_updates

def selected_models_noMF():
    fname_grid = [
        'entro_0',
        'g0_-2_entro0',
        'g0.25_-3_entro0',
        'g0.5_-3_entro0',
        'g0.8_-4_entro0'
        ]
    loss_weights_grid = [
        [0, 1E0, 1E0, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        {},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates

def selected_models(include_pos_sample_only=False):
    fname_grid = [
        #'mf',
        'entro_0',
        'g0_-2_entro0',
        'g0.25_-3_entro0',
        'g0.5_-3_entro0',
        'g0.8_-4_entro0'
        ]
    loss_weights_grid = [
        #[0, 0, 0, 1],
        [0, 1E0, 1E0, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        #{},
        {},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        ]

    if include_pos_sample_only:
        fname_grid.extend([
            'g0_-2',
            'g0.25_-3',
            'g0.5_-3',
            'g0.8_-3'
            ])
        loss_weights_grid.extend([
            [1E-2, 0, 0, 1],
            [1E-3, 0, 0, 1],
            [1E-3, 0, 0, 1],
            [1E-3, 0, 0, 1],
            ])
        param_updates.extend([
            {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
            {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
            {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
            {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
            ])

    return fname_grid, loss_weights_grid, param_updates

def full_grid():
    fname_grid = [
        'g0_-2_entro1_noMF',
        'g0.8_-4_entro1_noMF',
    
        'g0_-1',
        'g0_-2',
        'g0_-3',
    
        'g0.25_-1',
        'g0.25_-2',
        'g0.25_-3',
    
        'g0.5_-1',
        'g0.5_-2',
        'g0.5_-3',
    
        'g0.8_-1',
        'g0.8_-2',
        'g0.8_-3',
    
        'mf',
        'entro_0',
        'entro_1',
        'entro_2',
        'entro_3',
    
        'g0_-2_entro0',
        'g0_-2_entro1',
        'g0_-2_entro2',
        'g0_-2_entro3',
    
        'g0.25_-3_entro0',
        'g0.25_-3_entro1',
        'g0.25_-3_entro2',
        'g0.25_-3_entro3',
    
        'g0.5_-3_entro0',
        'g0.5_-3_entro1',
        'g0.5_-3_entro2',
        'g0.5_-3_entro3',
    
        'g0.8_-4_entro0',
        'g0.8_-4_entro1',
        'g0.8_-4_entro2',
        'g0.8_-4_entro3',
        ]
    
    loss_weights_grid = [
        [1E-2, 1E1, 1E1, 0],
        [1E-4, 1E1, 1E1, 0],
    
        [1E-1, 0, 0, 1],
        [1E-2, 0, 0, 1],
        [1E-3, 0, 0, 1],
    
        [1E-1, 0, 0, 1],
        [1E-2, 0, 0, 1],
        [1E-3, 0, 0, 1],
    
        [1E-1, 0, 0, 1],
        [1E-2, 0, 0, 1],
        [1E-3, 0, 0, 1],
    
        [1E-1, 0, 0, 1],
        [1E-2, 0, 0, 1],
        [1E-3, 0, 0, 1],
    
        [0, 0, 0, 1],
        [0, 1E0, 1E0, 1],
        [0, 1E1, 1E1, 1],
        [0, 1E2, 1E2, 1],
        [0, 1E3, 1E3, 1],
    
        [1E-2, 1E0, 1E0, 1],
        [1E-2, 1E1, 1E1, 1],
        [1E-2, 1E2, 1E2, 1],
        [1E-2, 1E3, 1E3, 1],
    
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E1, 1E1, 1],
        [1E-3, 1E2, 1E2, 1],
        [1E-3, 1E3, 1E3, 1],
    
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E1, 1E1, 1],
        [1E-3, 1E2, 1E2, 1],
        [1E-3, 1E3, 1E3, 1],
    
        [1E-4, 1E0, 1E0, 1],
        [1E-4, 1E1, 1E1, 1],
        [1E-4, 1E2, 1E2, 1],
        [1E-4, 1E3, 1E3, 1],
        ]
    
    param_updates = [
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
    
        {}, {}, {}, {}, {},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
    
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
        ]

    return fname_grid, loss_weights_grid, param_updates
