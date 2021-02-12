import utils
import random
import pickle
from tensorflow.python import keras
import numpy as np
import params as par
from random import randrange, gauss
        
def train_test_split(dataset, split=0.90):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


class Data:
    def __init__(self, sequences, token_eos, pad_token):
        self.token_eos = token_eos
        self.pad_token = pad_token
        self.sequences = sequences
        
    def __len__(self):
        return sum(len(s) for s in self.sequences.values()) 

    def batch(self, batch_size, length, mode='train'):
        batch_data = [
            self._get_seq(seq, length, mode)
            for seq in random.sample(self.sequences[mode], k=batch_size)
        ]
        return np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y
    
    def augment(self, ns):
        stretch_factor = gauss(1.0, 0.5)
        velocity_factor = gauss(1.0, 0.2)
        transpose = randrange(-5, 7)
        ns = stretch_note_sequence(ns, stretch_factor)
        for note in ns.notes:
            note.velocity = max(1, min(127, int(note.velocity * velocity_factor)))
        return transpose_note_sequence(ns, transpose, in_place=True)[0]

    def _get_seq(self, ns, max_length, mode):
        if mode == 'train':
            try:
                data = encode_note_sequence(midi_encoder, self.augment(ns))
            except BaseException as e:
                print(e)
                data = encode_note_sequence(midi_encoder, ns)
        else:
            data = encode_note_sequence(midi_encoder, ns)
            
        if max_length < len(data):
            start = random.randrange(0, len(data) - max_length)
            data = data[start:start + max_length]
        else:
            data = np.append(data, self.token_eos)
            while len(data) < max_length:
                data = np.append(data, self.pad_token)
        return data