from custom.layers import *
from custom.callback import *
import sys
from tensorflow.python import keras
import json
import tensorflow_probability as tfp
import random
import utils
from progress.bar import Bar
tf.executing_eagerly()


class MusicTransformerDecoder(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, pad_token=0):
        super(MusicTransformerDecoder, self).__init__()

        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist
        
        self.pad_token = pad_token

        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = keras.layers.Dense(self.vocab_size, activation=None, name='output')

        self._set_metrics()

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, training=None, eval=None, lookup_mask=None):
        decoder, w = self.Decoder(inputs, training=training, mask=lookup_mask)
        fc = self.fc(decoder)
        if training:
            return fc
        elif eval:
            return fc, w
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if self._debug:
            tf.print('sanity:\n', self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        x, y = self.__prepare_train_data(x, y)

        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, self.pad_token)

        if self.dist:
            predictions = self.__dist_train_step(
                x, y, look_ahead_mask, True)
        else:
            predictions = self.__train_step(x, y, look_ahead_mask, True)

        if self._debug:
            print('train step finished')
        result_metric = []

        if self.dist:
            loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, self.loss_value, None)
        else:
            loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(y, predictions).numpy())

        return [loss.numpy()]+result_metric

    # @tf.function
    def __dist_train_step(self, inp_tar, out_tar, lookup_mask, training):
        return self._distribution_strategy.experimental_run_v2(
            self.__train_step, args=(inp_tar, out_tar, lookup_mask, training))

    # @tf.function
    def __train_step(self, inp_tar, out_tar, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inputs=inp_tar, lookup_mask=lookup_mask, training=training
            )
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False):

        # x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, self.pad_token)
        predictions, w = self.call(
                x, lookup_mask=look_ahead_mask, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(y, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(y, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric, w

    def save(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'

        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")

    def sanity_check(self, x, y, mode='v', step=None):
        # mode: v -> vector, d -> dict
        # x, inp_tar, out_tar = self.__prepare_train_data(x, y)

        _, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, self.pad_token)
        predictions = self.call(
            x, lookup_mask=look_ahead_mask, training=False)

        if mode == 'v':
            tf.summary.image('vector', tf.expand_dims(predictions, -1), step)
            return predictions
        elif mode == 'd':
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            tf.summary.image('tokens', tf.argmax(predictions, -1), step)
            return tf.argmax(predictions, -1)

    def get_config(self):
        config = {}
        config['debug'] = self._debug
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        config['dist'] = self.dist
        return config
    
    def generate_beam(self, prior: list, length=2048, k=8, temperature=0.5):
        decode_array = prior
        decode_array = tf.constant([decode_array])
        seq_probs = [1.0]

        for i in range(min(self.max_seq, length)):
            if decode_array.shape[1] >= self.max_seq:
                break
            print('generating... {:.1f}% completed'.format((i/min(self.max_seq, length))*100), end="\r")
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array, self.pad_token)

            result = self.call(decode_array, lookup_mask=look_ahead_mask, training=False, eval=False)
            result = result[:,-1,:] / (temperature + 1e-6)
            probs, result_idx = tf.nn.top_k(result, k)

            result_array = []
            new_seq_probs = []
            for i in range(result_idx.shape[0]):
                for j in range(result_idx.shape[1]):
                    result_unit = tf.concat([decode_array[i], [result_idx[i, j]]], -1)
                    new_seq_probs.append(seq_probs[i] * probs[i, j].numpy())
                    result_array.append(result_unit.numpy())

            seq_probs, idx = tf.nn.top_k(new_seq_probs, k)
            decode_array = tf.constant([result_array[i] for i in idx])

        #         print(seq_probs)

            del look_ahead_mask

        seq_with_max_prob = np.argmax(seq_probs)
        decode_array = decode_array[seq_with_max_prob]
        return decode_array.numpy()

    def generate(self, prior: list, length=2048, temperature=0.0):
        decode_array = prior
        decode_array = tf.constant([decode_array])
        for i in range(min(self.max_seq, length)):
            if decode_array.shape[1] >= self.max_seq:
                break
            print('generating... {:.1f}% completed'.format((i/min(self.max_seq, length))*100), end="\r")
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array, self.pad_token)

            result, _ = self.call(decode_array, lookup_mask=look_ahead_mask, training=False, eval=True)
            result = result[:, -1] / (temperature + 1e-6)
            result = tf.nn.softmax(result)
            pdf = tfp.distributions.Categorical(probs=result)
            result = pdf.sample(1)
            result = tf.transpose(result, (1, 0))
            result = tf.cast(result, tf.int32)
            decode_array = tf.concat([decode_array, result], -1)
            del look_ahead_mask
        decode_array = decode_array[0]
        return decode_array.numpy()
    
    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']
        self.dist = config['dist']

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return

    @staticmethod
    def __prepare_train_data(x, y):
        # start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        # end_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_eos

        # # method with eos
        # out_tar = tf.concat([y[:, :-1], end_token], -1)
        # inp_tar = tf.concat([start_token, y[:, :-1]], -1)
        # x = tf.concat([start_token, x[:, 2:], end_token], -1)

        # method without eos
        # x = data.add_noise(x, rate=0.01)
        return x, y
