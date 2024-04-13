parameter_map = {}

### Grid search settings ###

def basic_test():
    fname_grid = [
        'mf0',
        'g0_-2_entro-1',
        ]
    loss_weights_grid = [
        [0,0,0,1E0],
        [1E-2, 1E-1, 1E-1,1E0]
        ]
    param_updates = [
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['basic_test'] = basic_test


def mf0():
    fname_grid = ['mf0']
    loss_weights_grid = [[0,0,0,1E0]]
    param_updates = [{}]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['mf0'] = mf0


def no_q():
    fname_grid = [
        'noQ_g0.5_-3_entro-3',
        'noQ_g0.5_-3_entro-2',
        'noQ_g0.5_-3_entro-1',
        ]
    loss_weights_grid = [
        [1E-3, 1E-3, 1E-3, 0],
        [1E-3, 1E-2, 1E-2, 0],
        [1E-3, 1E-1, 1E-1, 0],
        ]
    param_updates = [
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['no_1'] = no_q


def full_gridsearch():
    fname_grid = [
        'mf-1',
        'mf0',
        'mf1',

        'entro-2',
        'entro-1',
        'entro1',
        'entro0',
        'entro2',

        'g0_-2_entro-1',
        'g0.25_-2_entro-1',
        'g0.5_-2_entro-1',
        'g0.25_-3_entro-1',
        'g0.5_-3_entro-1',
        'g0.8_-3_entro-1',
        'g0.8_-4_entro-1',

        'g0_-2_entro-2',
        'g0.25_-2_entro-2',
        'g0.5_-2_entro-2',
        'g0.25_-3_entro-2',
        'g0.5_-3_entro-2',
        'g0.8_-3_entro-2',
        'g0.8_-4_entro-2',

        'g0_-2_entro0',
        'g0.25_-2_entro0',
        'g0.5_-2_entro0',
        'g0.25_-3_entro0',
        'g0.5_-3_entro0',
        'g0.8_-3_entro0',
        'g0.8_-4_entro0',

        'g0_-2_entro1',
        'g0.25_-2_entro1',
        'g0.5_-2_entro1',
        'g0.25_-3_entro1',
        'g0.5_-3_entro1',
        'g0.8_-3_entro1',
        'g0.8_-4_entro1',
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E-1],
        [0, 0, 0, 1E0],
        [0, 0, 0, 1E1],

        [0, 1E-2, 1E-2, 1],
        [0, 1E-1, 1E-1, 1],
        [0, 1E1, 1E1, 1],
        [0, 1E0, 1E0, 1],
        [0, 1E2, 1E2, 1],

        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-4, 1E-1, 1E-1, 1],

        [1E-2, 1E-2, 1E-2, 1],
        [1E-2, 1E-2, 1E-2, 1],
        [1E-2, 1E-2, 1E-2, 1],
        [1E-3, 1E-2, 1E-2, 1],
        [1E-3, 1E-2, 1E-2, 1],
        [1E-3, 1E-2, 1E-2, 1],
        [1E-4, 1E-2, 1E-2, 1],

        [1E-2, 1E0, 1E0, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],

        [1E-2, 1E1, 1E1, 1],
        [1E-2, 1E1, 1E1, 1],
        [1E-2, 1E1, 1E1, 1],
        [1E-3, 1E1, 1E1, 1],
        [1E-3, 1E1, 1E1, 1],
        [1E-3, 1E1, 1E1, 1],
        [1E-4, 1E1, 1E1, 1],
        ]
    param_updates = [
        {}, {}, {},
        {}, {}, {}, {}, {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},

        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},

        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},

        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['full_gridsearch'] = full_gridsearch


def mf_gridsearch():
    fname_grid = ['mf0', 'mf-1', 'mf1', 'mf2']
    loss_weights_grid = [
        [0,0,0,1E0],
        [0,0,0,1E-1],
        [0,0,0,1E1],
        [0,0,0,1E2],
        ]
    param_updates = [{}, {}, {}, {}]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['mf_gridsearch'] = mf_gridsearch


### Finalized model settings for experiments ###

def lineartrack():
    fname_grid = [
        'g0_-2_entro-2',
        'noq_g0_-2_entro-2',
        ]
    loss_weights_grid = [
        [1E-2, 1E-2, 1E-2, 1],
        [1E-2, 1E-2, 1E-2, 0],
        ]
    param_updates = [
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['lineartrack'] = lineartrack 


def selected_models_large_encoder():
    fname_grid = [
        'mf1', 
        'entro0',
        'g0_-2_entro-1',
        'g0.25_-2_entro0',
        'g0.5_-3_entro-1',
        'g0.8_-4_entro1'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E1],
        [0, 1E-0, 1E-0, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E-1, 1E-1, 1],
        [1E-4, 1E1, 1E1, 1],
        ]
    param_updates = [
        {},
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['selected_models_large_encoder'] = selected_models_large_encoder


def selected_models_large_q():
    fname_grid = [
        'mf0', 
        'entro2',
        'g0_-2_entro-1',
        'g0.25_-2_entro0',
        'g0.5_-2_entro0',
        'g0.8_-4_entro0'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E0],
        [0, 1E2, 1E2, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        {},
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['selected_models_large_q'] = selected_models_large_q


def selected_models_gridworld():
    fname_grid = [
        'mf1',
        'entro0',
        'g0_-2_entro1',
        'g0.25_-2_entro0',
        'g0.5_-2_entro1',
        'g0.8_-4_entro1'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E1],
        [0, 1E-0, 1E-0, 1],
        [1E-2, 1E1, 1E1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-2, 1E1, 1E1, 1],
        [1E-4, 1E1, 1E1, 1],
        ]
    param_updates = [
        {},
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['selected_models_gridworld'] = selected_models_gridworld


def selected_models_cifar():
    fname_grid = [
        'mf1',
        'entro0',
        'g0_-2_entro-1',
        'g0.25_-2_entro0',
        'g0.5_-3_entro0',
        'g0.8_-4_entro0'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E1],
        [0, 1E0, 1E0, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        {},
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['selected_models_cifar'] = selected_models_cifar


def selected_models_grid_shuffle():
    fname_grid = [
        'mf0',
        'entro0',
        'g0_-2_entro-1',
        'g0.25_-2_entro0',
        'g0.5_-3_entro0',
        'g0.8_-4_entro0'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1E0],
        [0, 1E0, 1E0, 1],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E0, 1E0, 1],
        [1E-3, 1E0, 1E0, 1],
        [1E-4, 1E0, 1E0, 1],
        ]
    param_updates = [
        {},
        {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['selected_models_grid_shuffle'] = selected_models_grid_shuffle


def altT():
    fname_grid = [
        'mf0',
        'g0_-2_entro-1',]
    loss_weights_grid = [
        [0, 0, 0, 1],
        [1E-2, 1E-1, 1E-1, 1],]
    param_updates = [
        {}, {}]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['altT'] = altT


def poort():
    fname_grid = [
        'mf0',
        'g0_-2_entro-3',
        'noq_g0_-1_entro-2',
        ]
    loss_weights_grid = [
        [0, 0, 0, 1],
        [1E-2, 1E-3, 1E-3, 1],
        [1E-1, 1E-2, 1E-2, 0],
        ]
    param_updates = [
        {}, {}, {}
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['poort'] = poort


def dswap():
    fname_grid = [
        'mf0',
        'entro-1',
        'noq_entro-1',
        'noq_g0_-2_entro-2',
        'noq_g0_-2_entro-1',
        'g0_-2_entro-1',
        'g0.25_-2_entro-2',
        'g0.5_-3_entro-2',
        'g0.8_-3_entro-2'
        ]
    loss_weights_grid = [
        [0, 0, 0, 1],
        [0, 1E-1, 1E-1, 1],
        [0, 1E-1, 1E-1, 0],
        [1E-2, 1E-2, 1E-2, 0],
        [1E-2, 1E-1, 1E-1, 0],
        [1E-2, 1E-1, 1E-1, 1],
        [1E-2, 1E-2, 1E-2, 1],
        [1E-3, 1E-2, 1E-2, 1],
        [1E-3, 1E-2, 1E-2, 1],
        ]
    param_updates = [
        {}, {}, {},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.25}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.5}},
        {'agent_args': {'pred_TD': True, 'pred_gamma': 0.8}},
        ]
    return fname_grid, loss_weights_grid, param_updates
parameter_map['dswap'] = dswap

