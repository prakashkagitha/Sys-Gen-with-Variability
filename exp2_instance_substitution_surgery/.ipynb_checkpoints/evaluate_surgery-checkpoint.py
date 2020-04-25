import sys
sys.path.append("../")

from seq2seq_lake_lstm import *
from scan_language import *

exp1_dir = "../exp1_emerging_systematic_generalization"
outputs_modifiers = np.load(f'{exp1_dir}/results/modifiers_outputs.npy', allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_train_data(n):  
    
    train_config = {"primitives": add_primitives(n, 'walk'),
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
    train_pairs = generate_data(train_config)

    test_config =  {"primitives": add_primitives(4, 'jump'),
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

def evaluation_surgery(pairs, model, randomly_retrieved_prims, seed=1, prims=None):
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
    
    
    matches, all_outputs = surgeryEvaluateRandomly(encoder, decoder, pairs, randomly_retrieved_prims, input_lang, output_lang, enc_max_length=enc_max_length, 
                                                   dec_max_length=dec_max_length, print_every=len(pairs)//3, seed=seed, prims=prims)
    return matches, all_outputs


if __name__ == '__main__':
    
    surgery_results = pd.DataFrame()
    all_outputs = []

#     for n in [3, 10, 20, 50, 100, 200, 300, 400, 500]:
    for n in [300]:
        train_prims = add_primitives(n, 'walk')
        test_prims = add_primitives(4, 'jump')
        all_prims = np.array(train_prims+test_prims)

        np.random.seed(1)

        train_pairs, test_pairs = get_train_data(n)
        for m_seed in range(5):
            accs1 = []
            accs2 = []
            outputs = []
            for seed in range(10):

                train_retrieved_prims = np.random.choice(train_prims, size=len(train_pairs))
                matches1, output1 = evaluation_surgery(train_pairs, f"{n}_prims_{m_seed}_seed", train_retrieved_prims, seed=seed, prims=all_prims)

                test_retrieved_prims = np.random.choice(train_prims, size=len(test_pairs))
                matches2, output2 = evaluation_surgery(test_pairs, f"{n}_prims_{m_seed}_seed", test_retrieved_prims, seed=seed, prims=all_prims)

                accs1.append(matches1/len(train_pairs))
                accs2.append(matches2/len(test_pairs))
                outputs.append((output1, output2))

            row1 = pd.Series(accs1, index=np.arange(10))
            row1.name =  f"{n}_prims_{m_seed}_seed_train"
            row2 = pd.Series(accs2, index=np.arange(10))
            row2.name =  f"{n}_prims_{m_seed}_seed_test"

            for row in [row1, row2]:
                surgery_results = surgery_results.append(row)
            all_outputs.append(outputs)
        
    surgery_results.to_csv("surgery_results_300prims.csv")