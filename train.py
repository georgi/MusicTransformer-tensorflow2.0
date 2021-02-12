from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import argparse
import datetime
import sys

tf.executing_eagerly()

from concurrent.futures import ThreadPoolExecutor

def run_training(epochs, dataset, batch_size, max_seq):
    batches_per_epoch = len(dataset) // batch_size
    num_eval_batches = 20

    def get_train_batch():
        return dataset.slide_seq2seq_batch(batch_size, max_seq, 'train')


    def get_eval_batch():
        return dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')


    with ThreadPoolExecutor(max_workers=1) as executor:
        for e in range(epochs):
            mt.reset_metrics()
            train_acc = []
            train_loss = []      
            valid_acc = []
            valid_loss = []

            futures = [
                executor.submit(get_train_batch)
                for b in range(batches_per_epoch)
            ]

            for future in futures:
                batch_x, batch_y = future.result()
                result_metrics = mt.train_on_batch(batch_x, batch_y)
                train_loss.append(result_metrics[0])
                train_acc.append(result_metrics[1])
                log_line = 'Train Loss: {:.3f}\tTrain Acc: {:.3f}'.format( 
                    np.mean(train_loss),
                    np.mean(train_acc),
                )
                print(log_line, end="\r")

            futures = [
                executor.submit(get_eval_batch)
                for b in range(num_eval_batches)
            ]

            for future in futures:
                eval_x, eval_y = future.result()
                eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
                valid_loss.append(eval_result_metrics[0])
                valid_acc.append(eval_result_metrics[1])

            mt.save(model_path)

            log_line = 'Epoch {}\tTrain Loss: {:.3f}\tTrain Acc: {:.3f}\tValid Loss: {:.3f}\tValid Acc: {:.3f}'.format(
                e,
                np.mean(train_loss),
                np.mean(train_acc), 
                np.mean(valid_loss),
                np.mean(valid_acc),
            )
            print(log_line)
            with open(os.path.join(model_path, 'metrics.log'), 'a') as f:
                print(log_line, file=f)

