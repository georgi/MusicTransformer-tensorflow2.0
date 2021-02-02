import pickle
import os
import re
import sys
import hashlib
from progress.bar import Bar
import tensorflow as tf
import utils
import params as par
from midi_processor.processor import encode_midi, decode_midi
from midi_processor import processor
import config
import random


def preprocess_midi_files_under(midi_root, save_dir, use_sustain=True):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(path)

        out_file = '{}/{}.pickle'.format(save_dir,path.split('/')[-1])
        if os.path.exists(out_file):
            continue

        try:
            data = encode_midi(path, use_sustain)
        except OSError:
            print("OSError")
        except ValueError:
            print("ValueError")
        except KeyError:
            print("KeyError")
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
        else:
            with open(out_file, 'wb') as f:
                pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])

