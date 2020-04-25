import sys
sys.path.append("../")

import os
import argparse
from random import shuffle

from seq2seq_lake_lstm import *
from scan_language import *

def train_grid(train_config1, train_config2, test_config, prim_repetition=1, seed=1, epochs=20, save_folder="trained_models"):
    print("Modules loaded successfully")

    train_pairs1 = generate_data(train_config1)
    train_pairs1 = train_pairs1 * prim_repetition
    train_pairs2 = generate_data(train_config2)
    train_pairs = train_pairs1 + train_pairs2
    shuffle(train_pairs)

    test_pairs = generate_data(test_config)

    all_pairs = train_pairs + test_pairs
    
    ENC_MAX_LENGTH = np.max([len(x.split()) for x in np.array(all_pairs)[:, 0]]) + 2
    DEC_MAX_LENGTH = np.max([len(x.split()) for x in np.array(all_pairs)[:, 1]])
    print(f"Max lengths {ENC_MAX_LENGTH} and {DEC_MAX_LENGTH}")

    input_lang, output_lang = read_data(all_pairs, 'Command', 'Action')
    print(random.choice(train_pairs))

    print("Initializing attention models for the same data")
    print("Device:", device)
    hidden_size = 100
    torch.manual_seed(seed)
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=1, dropout_p=0.1).to(device)
    torch.manual_seed(seed)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=ENC_MAX_LENGTH, dropout_p=0.1, n_layers=1).to(device)
    
    num_iters = len(train_pairs)*epochs

    print("Training of attention encoder-decoder model for productivity test has started...")
    trainIters(encoder, decoder, train_pairs, num_iters, input_lang, output_lang, print_every=num_iters//20, 
               enc_max_length=ENC_MAX_LENGTH, train_ratio=1,
               dec_max_length=DEC_MAX_LENGTH, learning_rate=0.001)
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = f"{save_folder}/{n}_prims_{seed}_seed_modifiers.pth"
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, save_path)

    print("Training completed and models saved")

    print("Loading saved best models")
    checkpoint = torch.load(save_path)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    
    encoder.eval()
    decoder.eval()

    print("Evaluating -- train ")
    matches, all_outputs = evaluateRandomly(encoder, decoder, train_pairs, input_lang, output_lang, ENC_MAX_LENGTH, DEC_MAX_LENGTH)
    train_acc = matches/len(train_pairs)
    print(train_acc)
    
    print("Evaluating -- test")
    matches, all_outputs = evaluateRandomly(encoder, decoder, test_pairs, input_lang, output_lang, ENC_MAX_LENGTH, DEC_MAX_LENGTH)
    test_acc = matches/len(test_pairs)
    print(test_acc)
    
    return (train_acc, test_acc), (all_outputs, save_path, input_lang, output_lang, ENC_MAX_LENGTH, DEC_MAX_LENGTH)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default="results", help="Output dir to save the generated dataset")
    
    args = parser.parse_args()
    out_dir = args.out_dir
    
    all_output_data = {} 
    acc_df = pd.DataFrame()
    for n in [3, 10, 20, 50, 100, 200, 300, 400, 500]:
        for seed in range(5):
            print("**"*50)
            print(f"For primtitives {n} and seed {seed}")

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

            train_config2 = {"primitives": add_primitives(n, 'walk'),
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

            accuracies, output_data = train_grid(train_config1, train_config2, test_config, seed=seed, epochs=25)
            all_output_data[f"{n}_prims_{seed}_seed"] = output_data

            train_acc, test_acc = accuracies
            row = pd.Series({"train_acc": train_acc, "test_acc":test_acc})
            row.name = f"{n}_prims_{seed}_seed"
            acc_df = acc_df.append(row)

    acc_df.to_csv(f"{out_dir}/modifiers_accuracies.csv")    
    np.save(f"{out_dir}/modifiers_outputs.npy", np.array([all_output_data]))