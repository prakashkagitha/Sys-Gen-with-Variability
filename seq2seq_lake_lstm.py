from __future__ import unicode_literals, print_function, division

from io import open
import unicodedata
import string
import random
import re
import time
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from fastprogress.fastprogress import progress_bar
from random import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            #print(word, self.n_words)
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
def readLangs(lang1, lang2, filename='SCAN/add_prim_split/tasks_train_addprim_jump.txt', reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').\
        read().strip().split('\n')
    
    
    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.replace("IN: ", "").split("OUT:")] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse=reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def read_data(pairs, lang1, lang2, reverse=False):
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout_p)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.randn(self.n_layers, 1, self.hidden_size, device=device), 
                torch.randn(self.n_layers, 1, self.hidden_size, device=device))
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.randn(self.n_layers, 1, self.hidden_size, device=device), 
                torch.randn(self.n_layers, 1, self.hidden_size, device=device))
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=None, n_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        hs, cs = hidden
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hs[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.randn(self.n_layers, 1, self.hidden_size, device=device), 
                torch.randn(self.n_layers, 1, self.hidden_size, device=device))
    
    
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def decoder_step(decoder, decoder_input, decoder_hidden, encoder_outputs):
    decoder_attention = None
    if encoder_outputs is not None:
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
    else:
        decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        
    return decoder_output, decoder_hidden, decoder_attention

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=None):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0
    #print(type(decoder))
    
    if isinstance(decoder, AttnDecoderRNN):
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    else:
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
        encoder_outputs = None

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_step(decoder,
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_step(decoder,
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, pairs, n_iters, input_lang, output_lang, train_ratio=0.8,
               print_every=1000, plot_every=100, learning_rate=0.01, enc_max_length=None, dec_max_length=None):
    start = time.time()
    val_accs = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    train_val_split = int(len(pairs)*train_ratio)
    train_pairs = pairs[:train_val_split]
    val_pairs = pairs[train_val_split:]
    
#     print(train_pairs)
#     print(np.array([tensorsFromPair(train_pairs[0], input_lang, output_lang)]))
    
    training_pairs = [tensorsFromPair(pair, input_lang, output_lang)
                                for pair in train_pairs]
    
    print(f"Got {len(training_pairs)} for train and {len(val_pairs)} for validation")
    print(len(training_pairs))
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[np.random.randint(0, len(training_pairs))]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=enc_max_length)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
                    
                    

def evaluate(encoder, decoder, sentence, input_lang, output_lang, enc_max_length=None, dec_max_length=None, seed=1):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        torch.manual_seed(seed)
        encoder_hidden = encoder.initHidden()
        
        if isinstance(decoder, AttnDecoderRNN):
            encoder_outputs = torch.zeros(enc_max_length, encoder.hidden_size, device=device)
            #print(input_length)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
        else:
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
            encoder_outputs = None

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(dec_max_length, enc_max_length)

        for di in range(dec_max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_step(decoder,
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
def evaluate_pairs(encoder, decoder, pairs, input_lang, output_lang, enc_max_length=None, dec_max_length=None):
    matches = 0
    for i, pair in enumerate(pairs):
        output_words, loss = evaluate(encoder, decoder, pair[0], input_lang, output_lang, enc_max_length=enc_max_length, dec_max_length=dec_max_length)
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('<EOS>', '').strip()
        if output_sentence==pair[1]:
            matches +=1
    return matches/len(pairs)
    
def evaluateRandomly(encoder, decoder, evaluation_pairs, input_lang, output_lang, enc_max_length=None, dec_max_length=None, print_every=1000, seed=1):
    total_loss = 0
    i= 0
    matches = 0
    all_outputs = []
    for pair in evaluation_pairs:

        output_words, loss = evaluate(encoder, decoder, pair[0], input_lang, output_lang, enc_max_length=enc_max_length, dec_max_length=dec_max_length, seed=seed)
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('<EOS>', '').strip()
        all_outputs.append(output_sentence)
        if output_sentence==pair[1]:
          matches +=1
        if i%print_every==0:  
          print('>', pair[0])
          print('=', pair[1])
          print('<', output_sentence)
          #print(loss)
          print('')
        #total_loss += loss
        i+=1
    print(matches)
    return matches, all_outputs

def replace_item_in_list(l, find_item, replace_item):
    return [replace_item if x==find_item else x for x in l]



def evaluate_surgery(encoder, decoder, current_sentence, retrieved_prim, input_lang, output_lang, enc_max_length=None, dec_max_length=None, 
                     seed=1, prims=None):
    with torch.no_grad():
        current_prim = prims[[prim in current_sentence for prim in prims]][0]
        #print(current_sentence, current_prim, retrieved_prim)
        retrieved_sentence = " ".join(replace_item_in_list(current_sentence.split(), current_prim, retrieved_prim))
        #print(retrieved_sentence)
        
        def get_hidden(sentence):
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            torch.manual_seed(seed)
            encoder_hidden = encoder.initHidden()

            if isinstance(decoder, AttnDecoderRNN):
                encoder_outputs = torch.zeros(enc_max_length, encoder.hidden_size, device=device)
                #print(input_length)
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                             encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]
            else:
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                             encoder_hidden)
                encoder_outputs = None
            return encoder_hidden, encoder_outputs
        
        
        #print(retrieved_sentence, current_prim, retrieved_prim)
        retrieved_sentence_hidden, encoder_outputs = get_hidden(retrieved_sentence)
        retrieved_prim_hidden, _ = get_hidden(retrieved_prim)
        current_prim_hidden, _ = get_hidden(current_prim)
        
        # change the encoder_hidden before giving it to decoder_hidden
        # loop over hidden state and cell state
        encoder_hidden_state = retrieved_sentence_hidden[0] - retrieved_prim_hidden[0] + current_prim_hidden[0]
        encoder_cell_state = retrieved_sentence_hidden[1] - retrieved_prim_hidden[1] + current_prim_hidden[1]
        
        encoder_hidden = (encoder_hidden_state, encoder_cell_state)
        
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        
        # as usual giving encoder_hidden to decoder_hidden
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(dec_max_length, enc_max_length)

        for di in range(dec_max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_step(decoder,
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def surgeryEvaluateRandomly(encoder, decoder, evaluation_pairs, randomly_retrieved_prims, input_lang, output_lang, enc_max_length=None, dec_max_length=None, 
                            print_every=1000, seed=1, prims=None):
    total_loss = 0
    i= 0
    matches = 0
    all_outputs = []
    for prim_i, pair in enumerate(evaluation_pairs):

        output_words = evaluate_surgery(encoder, decoder, pair[0], randomly_retrieved_prims[prim_i], input_lang, output_lang, 
                                              enc_max_length=enc_max_length, dec_max_length=dec_max_length, seed=seed, prims=prims)
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('<EOS>', '').strip()
        all_outputs.append(output_sentence)
        if output_sentence==pair[1]:
              matches +=1
        if i%print_every==0:  
          print('>', pair[0])
          print('=', pair[1])
          print('<', output_sentence)
          #print(loss)
          print('')
        #total_loss += loss
        i+=1
    print(matches)
    return matches, all_outputs