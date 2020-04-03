# 'generate_data' function generates the entire SCAN data with 'master_config' as input

import numpy as np

turn_dict = {'turn around left': 'I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT',
             'turn around right': 'I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT',
             'turn left': 'I_TURN_LEFT',
             'turn opposite left': 'I_TURN_LEFT I_TURN_LEFT',
             'turn opposite right': 'I_TURN_RIGHT I_TURN_RIGHT',
             'turn right': 'I_TURN_RIGHT'}

d_dict = {"left": "I_TURN_LEFT",
          "right": "I_TURN_RIGHT"}

def direction(action, direction):
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

def conjunction_concat_and(actions):
    return " ".join(actions)

def conjunction_concat_after(actions):
    return " ".join(actions[::-1])


functions_mapping = {"direction": direction, "opposite": opposite, "around": around, "twice": twice, "thrice": thrice, "and": conjunction_and, 
                     "after": conjunction_after, "and_concat": conjunction_concat_and, "after_concat": conjunction_concat_after}

master_config = {"primitives": ["look", "walk", "run", "jump"], "turn_commands":True, "directions": ["left", "right"], "modifiers": ["opposite", "around"],
                 "mod_directions": ["left", "right"], "multipliers": ["thrice","twice"], "conjunctions":["and", "after"], 
                 "concat_conjunctions": [], "concat_lengths": [[], []], "only_multipliers": False, "only_conjunctions": False,
                 "only_conjunction_concats": False}


def get_primitives(config):
    p_dict = {}
    for p in config["primitives"]:
        p_dict[p] = "I_"+ p.upper()
    return p_dict

def get_directions(config, x_dict, d_dict):
    direction_dict = {}
    for d in config["directions"]:
        for k,v in x_dict.items():
            action, direction = x_dict[k], d_dict[d]
            direction_dict[f"{k} {d}"] = functions_mapping["direction"](action, direction)
    return direction_dict

def get_modifiers(config, x_dict, d_dict):
    modifiers_dict = {}
    for m in config["modifiers"]:
        for d in config["mod_directions"]:
            for k,v in x_dict.items():
                action, direction = x_dict[k], d_dict[d]
                modifiers_dict[f"{k} {m} {d}"] = functions_mapping[m](action, direction)
    return modifiers_dict
                
def get_multipliers(config, x_dict):
    multipliers_dict = {}
    for m in config["multipliers"]:
        for k,v in x_dict.items():
            action = x_dict[k]
            multipliers_dict[f"{k} {m}"] = functions_mapping[m](action)
    return multipliers_dict
            
def get_conjunctions(config, x_dict):
    conjunctions_dict = {}
    for conjunction in config["conjunctions"]:
        for k1, v1 in x_dict.items():
            for k2, v2 in x_dict.items():
                action1, action2 = x_dict[k1], x_dict[k2]
                conjunctions_dict[f"{k1} {conjunction} {k2}"] = functions_mapping[conjunction](action1, action2)
    return conjunctions_dict
                
def get_conjunction_concats(config, x_dict):
    conj_concats_dict = {}
    for i, conjunction in enumerate(config["concat_conjunctions"]):
        for concat_lengths in config["concat_lengths"][i]:
            np.random.seed(seed)
            commands_list = np.random.choice(list(x_dict.keys()), size=(concat_counts, c))
            #keys = keys.reshape(concat_counts, c)
            for commands in commands_list:
                actions = [x_dict[command] for command in commands]
                and_dict[" {conjunction} ".join(commands)] = functions_mapping[f"{conjunction}_concat"](actions)
    return conj_concats_dict


def get_pairs_from_dict(x_dict):
    return [[k,v] for k, v in x_dict.items()]
    
def generate_data(**config):
    x_dict = {}
    
    primitives_dict = get_primitives(config)
    if config["keep_primitive_commands"]:    
        x_dict.update(primitives_dict)
    print(f"Total datapoints generated after adding primitives = {len(x_dict)}")
    
    if config["turn_commands"]:
        x_dict.update(turn_dict)
    print(f"Total datapoints generated after adding turn commands = {len(x_dict)}")
    
    directions_dict = get_directions(config, primitives_dict, d_dict)
    x_dict.update(directions_dict)
    print(f"Total datapoints generated after adding direction commands = {len(x_dict)}")
    
    modifiers_dict = get_modifiers(config, primitives_dict, d_dict)
    x_dict.update(modifiers_dict) 
    print(f"Total datapoints generated after adding modifiers = {len(x_dict)}")
    
    multipliers_dict = get_multipliers(config, x_dict)
    if config["only_multipliers"]:
        multipliers_pairs = get_pairs_from_dict(multipliers_dict)
        return multipliers_pairs
    x_dict.update(multipliers_dict)
    print(f"Total datapoints generated after adding multipliers = {len(x_dict)}")
    
    conjunctions_dict = get_conjunctions(config, x_dict)
    if config["only_conjunctions"]:
        conjunction_pairs = get_pairs_from_dict(conjunctions_dict)
        return conjunction_pairs
    x_dict.update(conjunctions_dict)
    print(f"Total datapoints generated after adding conjunctions = {len(x_dict)}")
    
    conjunction_concats_dict = get_conjunction_concats(config, x_dict)
    if config["only_conjunction_concats"]:
        conjunction_concats_pairs = get_pairs_from_dict(conjunction_concats_dict)
        return conjunction_concats_pairs
    x_dict.update(conjunction_concats_dict)
    print(f"Total datapoints generated with given config = {len(x_dict)}")
    
    return get_pairs_from_dict(x_dict)
    
# some ad-hoc post processing functions    

def transform_pairs(pairs):
    new_pairs = pairs.copy()
    for i in range(len(pairs)):
        new_pairs[i][0] = pairs[i][0].replace("_", " ")
    return new_pairs

def add_primitives(n, prim='walk'):
    return [f'{prim}_{i}' for i in range(n)]
