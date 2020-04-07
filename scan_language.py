###################
# 'generate_data' function generates the SCAN dataset (from Lake & Baroni (2017)) with 'master_config' dict as input. 
# The generating process is guided by a lot of parameters (see 'master_dict') to account for flexibility.
# These base utilities could generate the data for all the experiments in the paper.
###################


# Functions which represent syntactic transformations of directions/modifiers/multipliers/conjunctions.
# They take tokens representing action and direction to take and give out the sequence of actions.
# For example: opposite() takes ('I_WALK', 'I_TURN_LEFT') as input and gives "I_TURN_LEFT I_TURN_LEFT I_WALK"

def left(action, direction):
    return f"{direction} {action}"

def right(action, direction):
    return f"{direction} {action}"

def opposite(action, direction):
    return f"{direction} {direction} {action}"
    
def around(action, direction):
    return f"{direction} {action} {direction} {action} {direction} {action} {direction} {action}"

def twice(action):
    return f"{action} {action}"

def thrice(action):
    return f"{action} {action} {action}"

def conjunction_and(action1, action2):
    return f"{action1} {action2}"

def conjunction_after(action1, action2):
    return f"{action2} {action1}"


# Extra command, action pairs in SCAN dataset which are not associated with any primitive
turn_dict = {'turn around left': 'I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT',
             'turn around right': 'I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT',
             'turn left': 'I_TURN_LEFT',
             'turn opposite left': 'I_TURN_LEFT I_TURN_LEFT',
             'turn opposite right': 'I_TURN_RIGHT I_TURN_RIGHT',
             'turn right': 'I_TURN_RIGHT'}

# To transform command directions into action directions
d_dict = {"left": "I_TURN_LEFT",
          "right": "I_TURN_RIGHT"}

# A dict mapping actions tokens to functions we have to use to generate actions
functions_mapping = {"left": left,
                     "right": right,
                     "opposite": opposite, 
                     "around": around, 
                     "twice": twice, 
                     "thrice": thrice, 
                     "and": conjunction_and, 
                     "after": conjunction_after}


def get_primitives(config):
    "Generates primitive command, action pairs"
    
    p_dict = {}
    for p in config["primitives"]:
        p_dict[p] = "I_"+ p.upper()
    return p_dict

def get_directions(config, x_dict, d_dict):
    "Takes current x_dict which has some command, action pairs and applies directions to them"
    
    direction_dict = {}
    for d in config["directions"]:
        for k,v in x_dict.items():
            action, direction = x_dict[k], d_dict[d]
            direction_dict[f"{k} {d}"] = functions_mapping[d](action, direction)
    return direction_dict

def get_modifiers(config, x_dict, d_dict):
    "Takes current x_dict which has some command, action pairs and applies modifiers to them"
    
    modifiers_dict = {}
    for m in config["modifiers"]:
        for d in config["modifier_directions"]:
            for k,v in x_dict.items():
                action, direction = x_dict[k], d_dict[d]
                modifiers_dict[f"{k} {m} {d}"] = functions_mapping[m](action, direction)
    return modifiers_dict
                
def get_multipliers(config, x_dict):
    "Takes current x_dict which has some command, action pairs and applies multipliers to them"
    
    multipliers_dict = {}
    for m in config["multipliers"]:
        for k,v in x_dict.items():
            action = x_dict[k]
            multipliers_dict[f"{k} {m}"] = functions_mapping[m](action)
    return multipliers_dict
            
def get_conjunctions(config, x_dict):
    "Takes current x_dict which has some command, action pairs and applies conjunctions to them"
    
    conjunctions_dict = {}
    for conjunction in config["conjunctions"]:
        for k1, v1 in x_dict.items():
            for k2, v2 in x_dict.items():
                action1, action2 = x_dict[k1], x_dict[k2]
                conjunctions_dict[f"{k1} {conjunction} {k2}"] = functions_mapping[conjunction](action1, action2)
    return conjunctions_dict


def get_pairs_from_dict(x_dict):
    "Making a dict list of lists"
    
    return [[k,v] for k, v in x_dict.items()]

def add_primitives(n, prim='walk'):
    "Outputs a list of synthetic primitives like walk_2 or walk_300"
    
    return [f'{prim}_{i}' for i in range(n)]
    

# The configuration which generates the complete SCAN dataset when given to generate_data()
master_config = {"primitives": ["look", "walk", "run", "jump"],
                 "keep_primitive_commands": True,
                 "keep_turn_commands":True, 
                 "directions": ["left", "right"], 
                 "modifiers": ["opposite", "around"],
                 "modifier_directions": ["left", "right"], 
                 "multipliers": ["twice", "thrice"], 
                 "conjunctions":["and", "after"], 
                 "only_modifiers": False,
                 "only_multipliers": False, 
                 "only_conjunctions": False}


def generate_data(config):
    """ 
    Takes a config dict and generates SCAN-like dataset
    
    Args:
        config (dict): contains the following parameters to guide the generation process
            primitives (list): contains primitives from which to generate dataset
            keep_primitive_commands (bool): whether to include ('walk', 'I_WALK') style datapoints
                which are not necessarity when building custom test set.
            keep_turn_commands (bool): whether to keep turn commands which are in SCAN dataset but are excluded in
                some of analyses done of SCAN dataset before like in Loula et al. (2018).
            directions (list): contains direction commands to be applied
            modifiers (list): contains modifier commands to be applied
            modifier_directions (list): contains directions with which modifiers are used (as in 'walk opposite(modifier) left(direction)')
            multipliers (list): contains multiplier commands to be applied
            conjunctions (list): contains conjunction commands to be applied
            only_modifiers (bool): whether to output only datapoints which are produced by modifiers
            only_multipliers (bool): whether to output only datapoints which are produced by multipliers
            only_conjunctions (bool): whether to output only datapoints which are produced by conjunctions
            
    Returns:
        list of lists: a dataset generateed from the given parameters
    """
    
    x_dict = {}
    
    primitives_dict = get_primitives(config)
    if config["keep_primitive_commands"]:    
        x_dict.update(primitives_dict)
    print(f"Total datapoints generated after adding primitives = {len(x_dict)}")
    
    if config["keep_turn_commands"]:
        x_dict.update(turn_dict)
    print(f"Total datapoints generated after adding turn commands = {len(x_dict)}")
    
    directions_dict = get_directions(config, primitives_dict, d_dict)
    x_dict.update(directions_dict)
    print(f"Total datapoints generated after adding direction commands = {len(x_dict)}")
    
    modifiers_dict = get_modifiers(config, primitives_dict, d_dict)
    if config["only_modifiers"]:
        return get_pairs_from_dict(modifiers_dict)
    x_dict.update(modifiers_dict) 
    print(f"Total datapoints generated after adding modifiers = {len(x_dict)}")
    
    multipliers_dict = get_multipliers(config, x_dict)
    if config["only_multipliers"]:
        return get_pairs_from_dict(multipliers_dict)
    x_dict.update(multipliers_dict)
    print(f"Total datapoints generated after adding multipliers = {len(x_dict)}")
    
    conjunctions_dict = get_conjunctions(config, x_dict)
    if config["only_conjunctions"]:
        return get_pairs_from_dict(conjunctions_dict)
    x_dict.update(conjunctions_dict)
    print(f"Total datapoints generated after adding conjunctions = {len(x_dict)}")
    
    return get_pairs_from_dict(x_dict)
