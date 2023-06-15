
def params_A():
    """
    Chosen for gridworld8x8_shuffobs, internal dim 16.
    Criteria: ABCDEF cannot be goal locations. ABC and DEF
    must be a minimum of 5 actions away from each other.
    The chosen_unit should respond similarly strongly to ABC
    and low to DEF.
    """

    params = {}
    params['_0'] = {
        'a': (), 'b': (), 'c': (),
        'd': (), 'e': (), 'f': (),
        'bf_action':, 'fb_action':, 'ec_action': 'ce_action':,
        'chosen_unit': , 'goal_loc': ()}

    return params
