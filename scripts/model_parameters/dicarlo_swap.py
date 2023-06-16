
def param_set_1():
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

    # Rewardless and negative sampling only
    params['noQ_entro0_0'] = {
        'a': (1,3), 'b': (2,3), 'c': (3,3), 'd': (7,8), 'e': (8,8), 'f': (8,7),
        'chosen_unit': 7, 'goal_loc': (7,2)}
    params['noQ_entro0_1'] = {
        'a': (2,2), 'b': (2,3), 'c': (2,4), 'd': (7,4), 'e': (7,5), 'f': (7,6),
        'chosen_unit': 3, 'goal_loc': (5,6)}
    params['noQ_entro0_2'] = {
        'a': (1,6), 'b': (1,7), 'c': (1,8), 'd': (5,1), 'e': (6,1), 'f': (7,1),
        'chosen_unit': 13, 'goal_loc': (3,6)}
    params['noQ_entro0_3'] = {
        'a': (6,3), 'b': (7,3), 'c': (8,3), 'd': (1,8), 'e': (2,8), 'f': (3,8),
        'chosen_unit': 2, 'goal_loc': (2,7)}
    params['noQ_entro0_4'] = {
        'a': (3,4), 'b': (4,4), 'c': (5,4), 'd': (1,7), 'e': (1,8), 'f': (2,8),
        'chosen_unit': 14, 'goal_loc': (8,7)}

    # Rewardless and one-step positive sampling only
    params['noQ_g0_-2_entro0_0'] = {
        'a': (4,3), 'b': (5,3), 'c': (6,3), 'd': (7,8), 'e': (8,8), 'f': (8,7),
        'chosen_unit': 5, 'goal_loc': (7,2)}
    params['noQ_g0_-2_entro0_1'] = {
        'a': (2,2), 'b': (2,3), 'c': (2,4), 'd': (7,8), 'e': (7,7), 'f': (8,7),
        'chosen_unit': 0, 'goal_loc': (5,6)}
    params['noQ_g0_-2_entro0_2'] = {
        'a': (3,3), 'b': (4,3), 'c': (4,4), 'd': (7,8), 'e': (8,8), 'f': (8,7),
        'chosen_unit': 8, 'goal_loc': (3,6)}
    params['noQ_g0_-2_entro0_3'] = {
        'a': (3,7), 'b': (3,6), 'c': (4,6), 'd': (7,1), 'e': (7,2), 'f': (7,3),
        'chosen_unit': 7, 'goal_loc': (2,7)}
    params['noQ_g0_-2_entro0_4'] = {
        'a': (8,3), 'b': (8,4), 'c': (8,5), 'd': (1,6), 'e': (1,7), 'f': (1,8),
        'chosen_unit': 12, 'goal_loc': (8,7)}

    # Negative sampling task only
    params['entro0_0'] = {
        'a': (7,4), 'b': (6,4), 'c': (6,5), 'd': (2,1), 'e': (2,2), 'f': (3,2),
        'chosen_unit': 15, 'goal_loc': (7,2)}
    params['entro0_1'] = {
        'a': (4,5), 'b': (5,5), 'c': (6,5), 'd': (1,2), 'e': (1,1), 'f': (2,1),
        'chosen_unit': 0, 'goal_loc': (5,6)}
    params['entro0_2'] = {
        'a': (1,7), 'b': (2,7), 'c': (2,6), 'd': (6,3), 'e': (7,3), 'f': (8,3),
        'chosen_unit': 7, 'goal_loc': (3,6)}
    params['entro0_3'] = {
        'a': (1,5), 'b': (1,6), 'c': (2,6), 'd': (6,2), 'e': (7,2), 'f': (8,2),
        'chosen_unit': 10, 'goal_loc': (2,7)}
    params['entro0_4'] = {
        'a': (1,3), 'b': (1,4), 'c': (2,4), 'd': (7,2), 'e': (7,3), 'f': (7,4),
        'chosen_unit': 1, 'goal_loc': (8,7)}

    # Negative and one-step positive sampling task
    params['g0_-2_entro-1_0'] = {
        'a': (1,2), 'b': (1,3), 'c': (2,3), 'd': (6,6), 'e': (7,6), 'f': (7,5),
        'chosen_unit': 4, 'goal_loc': (7,2)}
    params['g0_-2_entro-1_1'] = {
        'a': (2,3), 'b': (2,2), 'c': (3,2), 'd': (6,6), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 3, 'goal_loc': (5,6)}
    params['g0_-2_entro-1_2'] = {
        'a': (2,4), 'b': (3,4), 'c': (4,4), 'd': (7,7), 'e': (8,7), 'f': (8,8),
        'chosen_unit': 14, 'goal_loc': (3,6)}
    params['g0_-2_entro-1_3'] = {
        'a': (1,2), 'b': (1,1), 'c': (2,1), 'd': (7,5), 'e': (7,6), 'f': (7,7),
        'chosen_unit': 1, 'goal_loc': (2,7)}
    params['g0_-2_entro-1_4'] = {
        'a': (3,7), 'b': (4,7), 'c': (5,7), 'd': (8,1), 'e': (8,2), 'f': (8,3),
        'chosen_unit': 12, 'goal_loc': (8,7)}

    return params
