import sys
sys.path.append("../")

from seq2seq_lake_lstm import *
from scan_language import *

exp1_dir = "../exp1_emerging_systematic_generalization"
outputs_modifiers = np.load(f'{exp1_dir}/results/modifiers_outputs.npy', allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_compound_pairs(base_directions=["left"], compound_directions=["left"]):
    
    base_config =  {"primitives": add_primitives(4, 'jump'),
                    "keep_primitive_commands": False,
                    "keep_turn_commands":False, 
                    "directions": base_directions, 
                    "modifiers": [],
                    "modifier_directions": [], 
                    "multipliers": [], 
                    "conjunctions":[], 
                    "only_modifiers": False,
                    "only_multipliers": False, 
                    "only_conjunctions": False}
    
    base_pairs = generate_data(base_config)
    base_dict = {k:v for (k,v) in base_pairs}

    compound_config = {"primitives": add_primitives(4, 'jump'),
                       "keep_primitive_commands": False,
                       "keep_turn_commands":False, 
                       "directions": compound_directions, 
                       "modifiers": [],
                       "modifier_directions": [], 
                       "multipliers": [], 
                       "conjunctions":[], 
                       "only_modifiers": False,
                       "only_multipliers": False, 
                       "only_conjunctions": False}
    
    compound_dict = get_directions(compound_config, base_dict, d_dict)
    compound_pairs = [[k,v] for k,v in compound_dict.items()]
    
    print(f"Generated {len(compound_pairs)} compound pairs with base directions {base_directions} and compound directions {compound_directions}")
    return compound_pairs

def evaluate_compound(pairs, model, seed=1):
    output_tuple = outputs_modifiers[0][model]
    model_path = exp1_dir + "/" + "models/" + output_tuple[1].split("/")[-1]
    
    input_lang = output_tuple[2]
    output_lang = output_tuple[3]
    enc_max_length = output_tuple[4]
    dec_max_length = output_tuple[5]
    
    if device == torch.device("cpu"):
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path)
    
    hidden_size =100
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=1, dropout_p=0.1).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=enc_max_length, dropout_p=0.1, n_layers=1).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    
    matches, all_outputs = evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, enc_max_length=enc_max_length, dec_max_length=dec_max_length,
                                            print_every=5, seed=seed)
    return matches, all_outputs


def save_dataset(pairs, fname="dataset.txt"):
    
    with open(fname, "w") as text_file:
        for pair in pairs:
            text_file.write(f"IN: {pair[0]} OUT: {pair[1]}\n")

if __name__ == '__main__':
    
    compound_pairs1 = get_compound_pairs(base_directions=["left"], compound_directions=["left"]) + \
                      get_compound_pairs(base_directions=["right"], compound_directions=["right"])
    save_dataset(compound_pairs1, fname="compound_variables_testset1.txt")
    
    compound_pairs2 = get_compound_pairs(base_directions=["left"], compound_directions=["right"]) + \
                      get_compound_pairs(base_directions=["right"], compound_directions=["left"])
    save_dataset(compound_pairs2, fname="compound_variables_testset2.txt")

    compound_results = pd.DataFrame()
    all_outputs = []

#     for n in [3, 10, 20, 50, 100, 200, 300, 400, 500]:
    for n in [300]:
        for m_seed in [0, 1, 2, 3, 4]:
            accs1 = []
            accs2 = []
            outputs = []
            for seed in range(10):
                matches1, output1 = evaluate_compound(compound_pairs1, f"{n}_prims_{m_seed}_seed", seed=seed)
                matches2, output2 = evaluate_compound(compound_pairs2, f"{n}_prims_{m_seed}_seed", seed=seed)
                accs1.append(matches1/len(compound_pairs1))
                accs2.append(matches2/len(compound_pairs2))
                outputs.append((output1, output2))

            row1 = pd.Series(accs1, index=np.arange(10))
            row1.name =  f"{n}_prims_{m_seed}_seed_test_1"
            row2 = pd.Series(accs2, index=np.arange(10))
            row2.name =  f"{n}_prims_{m_seed}_seed_test_2"

            for row in [row1, row2]:
                compound_results = compound_results.append(row)
            all_outputs.append(outputs)
    
    compound_results.to_csv("compound_results_300prims.csv")