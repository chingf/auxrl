def postbug_contig():
    """
    Chosen for gridworld8x8_shuffobs, internal dim 16.
    Criteria: ABCDEF cannot be goal locations. ABC and DEF
    must be a minimum of 5 actions away from each other.
    The chosen_unit should respond similarly strongly to ABC
    and low to DEF.

    bf_action is the normal b -> c action
    fb_action is the normal f -> e action
    ec_action is the normal e -> f action
    ce_action is the normal c -> b action
    """    
    
    params = {}

    # No reward learning; positive-sampling
    params['noq_g0_-2_entro-1_0'] = {
        'a': (1,2), 'b': (1,1), 'c': (2,1), 'd': (8,7), 'e': (8,8), 'f': (7,8),
        'chosen_unit': 2}
    params['noq_g0_-2_entro-1_1'] = {
        'a': (1,7), 'b': (2,7), 'c': (2,8), 'd': (1,1), 'e': (2,1), 'f': (2,2),
        'chosen_unit': 3}
    params['noq_g0_-2_entro-1_2'] = {
        'a': (3,7), 'b': (3,8), 'c': (2,8), 'd': (5,2), 'e': (6,2), 'f': (7,2),
        'chosen_unit': 8}
    params['noq_g0_-2_entro-1_3'] = {
        'a': (1,2), 'b': (2,2), 'c': (2,1), 'd': (6,7), 'e': (6,6), 'f': (7,6),
        'chosen_unit': 9}
    params['noq_g0_-2_entro-1_4'] = {
        'a': (6,1), 'b': (7,1), 'c': (8,1), 'd': (2,7), 'e': (3,7), 'f': (4,7),
        'chosen_unit': 7}
    params['noq_g0_-2_entro-1_5'] = {
        'a': (6,1), 'b': (7,1), 'c': (7,2), 'd': (1,6), 'e': (1,7), 'f': (2,7),
        'chosen_unit': 4}
    params['noq_g0_-2_entro-1_6'] = {
        'a': (8,1), 'b': (8,2), 'c': (8,3), 'd': (3,7), 'e': (3,8), 'f': (4,8),
        'chosen_unit': 6}
    params['noq_g0_-2_entro-1_7'] = {
        'a': (1,3), 'b': (2,3), 'c': (2,2), 'd': (6,7), 'e': (7,7), 'f': (8,7),
        'chosen_unit': 5}
    params['noq_g0_-2_entro-1_8'] = {
        'a': (7,1), 'b': (8,1), 'c': (8,2), 'd': (1,6), 'e': (2,6), 'f': (2,7),
        'chosen_unit': 2}
    params['noq_g0_-2_entro-1_9'] = {
        'a': (1,4), 'b': (1,5), 'c': (1,6), 'd': (7,5), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 9}
    params['noq_g0_-2_entro-1_10'] = {
        'a': (7,1), 'b': (8,1), 'c': (8,2), 'd': (2,4), 'e': (2,5), 'f': (2,6),
        'chosen_unit': 7}
    params['noq_g0_-2_entro-1_11'] = {
        'a': (7,7), 'b': (8,7), 'c': (8,6), 'd': (1,3), 'e': (2,3), 'f': (2,2),
        'chosen_unit': 9}
    params['noq_g0_-2_entro-1_12'] = {
        'a': (5,1), 'b': (6,1), 'c': (7,1), 'd': (5,8), 'e': (6,8), 'f': (7,8),
        'chosen_unit': 0}
    params['noq_g0_-2_entro-1_13'] = {
        'a': (7,7), 'b': (7,8), 'c': (6,8), 'd': (1,7), 'e': (2,7), 'f': (2,6),
        'chosen_unit': 14}
    params['noq_g0_-2_entro-1_14'] = {
        'a': (1,2), 'b': (1,3), 'c': (1,4), 'd': (7,5), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 8}

    # Reward learning; positive-sampling
    params['g0_-2_entro-1_0'] = {
        'a': (1,8), 'b': (2,8), 'c': (3,8), 'd': (8,4), 'e': (8,5), 'f': (8,6),
        'chosen_unit': 5}
    params['g0_-2_entro-1_1'] = {
        'a': (4,5), 'b': (4,6), 'c': (4,7), 'd': (7,2), 'e': (7,3), 'f': (7,4),
        'chosen_unit': 1}
    params['g0_-2_entro-1_2'] = {
        'a': (1,6), 'b': (1,7), 'c': (1,8), 'd': (6,1), 'e': (7,1), 'f': (8,1),
        'chosen_unit': 15}
    params['g0_-2_entro-1_3'] = {
        'a': (8,3), 'b': (8,2), 'c': (8,1), 'd': (4,7), 'e': (5,7), 'f': (5,8),
        'chosen_unit': 3}
    params['g0_-2_entro-1_4'] = {
        'a': (1,3), 'b': (1,2), 'c': (2,2), 'd': (7,4), 'e': (7,5), 'f': (7,6),
        'chosen_unit': 13}
    params['g0_-2_entro-1_5'] = {
        'a': (1,3), 'b': (1,2), 'c': (2,2), 'd': (7,5), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 2}
    params['g0_-2_entro-1_6'] = {
        'a': (6,4), 'b': (6,3), 'c': (7,3), 'd': (2,7), 'e': (3,7), 'f': (4,7),
        'chosen_unit': 7}
    params['g0_-2_entro-1_7'] = {
        'a': (3,8), 'b': (4,8), 'c': (4,7), 'd': (7,4), 'e': (7,5), 'f': (8,5),
        'chosen_unit': 14}
    params['g0_-2_entro-1_8'] = {
        'a': (5,8), 'b': (6,8), 'c': (7,8), 'd': (1,3), 'e': (1,2), 'f': (2,2),
        'chosen_unit': 8}
    params['g0_-2_entro-1_9'] = {
        'a': (1,6), 'b': (1,7), 'c': (1,8), 'd': (7,2), 'e': (7,1), 'f': (8,1),
        'chosen_unit': 5}
    params['g0_-2_entro-1_10'] = {
        'a': (6,7), 'b': (6,8), 'c': (7,8), 'd': (2,4), 'e': (2,5), 'f': (2,6),
        'chosen_unit': 2}
    params['g0_-2_entro-1_11'] = {
        'a': (2,8), 'b': (3,8), 'c': (3,7), 'd': (5,1), 'e': (6,1), 'f': (6,2),
        'chosen_unit': 0}
    params['g0_-2_entro-1_12'] = {
        'a': (4,4), 'b': (3,4), 'c': (3,5), 'd': (7,4), 'e': (8,4), 'f': (8,5),
        'chosen_unit': 12}
    params['g0_-2_entro-1_13'] = {
        'a': (1,2), 'b': (1,3), 'c': (2,3), 'd': (4,7), 'e': (5,7), 'f': (6,7),
        'chosen_unit': 6}
    params['g0_-2_entro-1_14'] = {
        'a': (6,2), 'b': (6,1), 'c': (7,1), 'd': (3,3), 'e': (3,4), 'f': (4,4),
        'chosen_unit': 12}
    
    # Reward learning; negative-sampling
    params['entro-1_0'] = {
        'a': (1,6), 'b': (2,6), 'c': (2,7), 'd': (6,1), 'e': (6,2), 'f': (6,3),
        'chosen_unit': 7}
    params['entro-1_1'] = {
        'a': (5,5), 'b': (5,4), 'c': (6,4), 'd': (1,7), 'e': (2,7), 'f': (2,8),
        'chosen_unit': 7}
    params['entro-1_2'] = {
        'a': (4,4), 'b': (5,4), 'c': (5,3), 'd': (7,2), 'e': (8,2), 'f': (8,1),
        'chosen_unit': 14}
    params['entro-1_3'] = {
        'a': (1,4), 'b': (1,3), 'c': (2,3), 'd': (7,4), 'e': (7,5), 'f': (7,6),
        'chosen_unit': 12}
    params['entro-1_4'] = {
        'a': (7,7), 'b': (7,6), 'c': (7,5), 'd': (2,3), 'e': (2,2), 'f': (3,2),
        'chosen_unit': 2}
    params['entro-1_5'] = {
        'a': (6,1), 'b': (7,1), 'c': (7,2), 'd': (3,5), 'e': (4,5), 'f': (4,6),
        'chosen_unit': 13}
    params['entro-1_6'] = {
        'a': (3,2), 'b': (3,3), 'c': (4,3), 'd': (7,8), 'e': (7,7), 'f': (8,7),
        'chosen_unit': 0}
    params['entro-1_7'] = {
        'a': (6,3), 'b': (6,4), 'c': (6,5), 'd': (1,8), 'e': (2,8), 'f': (3,8),
        'chosen_unit': 13}
    params['entro-1_8'] = {
        'a': (5,1), 'b': (6,1), 'c': (7,1), 'd': (3,7), 'e': (3,8), 'f': (4,8),
        'chosen_unit': 1}
    params['entro-1_9'] = {
        'a': (8,8), 'b': (7,8), 'c': (7,7), 'd': (2,2), 'e': (2,3), 'f': (2,4),
        'chosen_unit': 2}
    params['entro-1_10'] = {
        'a': (2,3), 'b': (2,4), 'c': (3,4), 'd': (7,5), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 1}
    params['entro-1_11'] = {
        'a': (1,7), 'b': (1,8), 'c': (2,8), 'd': (4,1), 'e': (4,2), 'f': (4,3),
        'chosen_unit': 14}
    params['entro-1_12'] = {
        'a': (2,2), 'b': (2,3), 'c': (2,4), 'd': (7,2), 'e': (7,3), 'f': (7,4),
        'chosen_unit': 9}
    params['entro-1_13'] = {
        'a': (8,5), 'b': (7,5), 'c': (7,4), 'd': (2,2), 'e': (2,3), 'f': (2,4),
        'chosen_unit': 0}
    params['entro-1_14'] = {
        'a': (1,3), 'b': (1,4), 'c': (1,5), 'd': (5,1), 'e': (5,2), 'f': (6,2),
        'chosen_unit': 15}

    return params
