import sys
sys.path.append("../")

import os
import argparse

from scan_language import *


def gen_train_test(n_prims):
    
    train_config1 = {"primitives": add_primitives(4, 'jump'),
                    "keep_primitive_commands": True,
                    "keep_turn_commands":False, 
                    "directions": [], 
                    "modifiers": [],
                    "modifier_directions": [], 
                    "multipliers": [], 
                    "conjunctions":[], 
                    "only_modifiers": False,
                    "only_multipliers": False, 
                    "only_conjunctions": False}
    
    train_config2 = {"primitives": add_primitives(n_prims, 'walk'),
                    "keep_primitive_commands": True,
                    "keep_turn_commands":False, 
                    "directions": ["left", "right"], 
                    "modifiers": ["opposite", "around"],
                    "modifier_directions": ["left", "right"], 
                    "multipliers": [], 
                    "conjunctions":[], 
                    "only_modifiers": False,
                    "only_multipliers": False, 
                    "only_conjunctions": False}
    
    train_pairs1 = generate_data(train_config1)
    train_pairs2 = generate_data(train_config2)
    train_pairs = train_pairs1 + train_pairs2

    test_config =   {"primitives": add_primitives(4, 'jump'),
                    "keep_primitive_commands": False,
                    "keep_turn_commands":False, 
                    "directions": ["left", "right"], 
                    "modifiers": ["opposite", "around"],
                    "modifier_directions": ["left", "right"], 
                    "multipliers": [], 
                    "conjunctions":[], 
                    "only_modifiers": False,
                    "only_multipliers": False, 
                    "only_conjunctions": False}
    
    test_pairs = generate_data(test_config)
    
    return train_pairs, test_pairs


def save_dataset(pairs, fname="dataset.txt"):
    
    with open(fname, "w") as text_file:
        for pair in pairs:
            text_file.write(f"IN: {pair[0]} OUT: {pair[1]}\n")

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_prims', type=int, default=300, help='Number of distinct primitives in the dataset')
    parser.add_argument('--fname', type=str, default="data", help='File name for the generated dataset')
    parser.add_argument('--out_dir', type=str, default="dataset", help="Output dir to save the generated dataset")
    
    args = parser.parse_args()
    n_prims = args.n_prims
    fname = args.fname
    out_dir = args.out_dir
    
    train_pairs, test_pairs = gen_train_test(n_prims)
    print(f"    Number of datapoints in train set: {len(train_pairs)}")
    print(f"    Number of datapoints in test set: {len(test_pairs)}")
    
    os.makedirs(out_dir, exist_ok=True)
    train_fname = f"{out_dir}/{fname}_train.txt"
    save_dataset(train_pairs, fname=train_fname)
    print(f"    Saved train set at '{train_fname}'")
    
    test_fname = f"{out_dir}/{fname}_test.txt"
    save_dataset(test_pairs, fname=test_fname)
    print(f"    Saved test set at '{test_fname}'")