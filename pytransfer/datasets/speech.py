# # -*- coding: utf-8 -*-
# This file is adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.

import os.path
import sys
import tarfile
import glob

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib
import torch.utils.data as data
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

from pytransfer.datasets.base import DomainDatasetBase

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
DATA_URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
ROOT_DIR = os.path.expanduser('~/.torch/datasets')
DATA_DIR = os.path.join(ROOT_DIR, 'speech')
BACKGROUND_VOLUME = 0.1
TIME_SHIFT_MS = 100.0
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
DCT_COEFFICIENT_COUNT = 40
WANTED_WORDS = 'yes,no,up,down,left,right,on,off,stop,go,cat,dog'


def prepare_model_settings(sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'sample_rate': sample_rate,
    }


class _SingleSpeech(data.Dataset):
    all_domain_key = ['doing_the_dishes_1', 'doing_the_dishes_2',
                      'dude_miaowing_1', 'dude_miaowing_2',
                      'exercise_bike_1', 'exercise_bike_2',
                      'running_tap_1', 'running_tap_2']

    data_dir = DATA_DIR
    wanted_words = WANTED_WORDS.split(',')
    word_to_index = {k: i for i, k in enumerate(wanted_words)}
    num_classes = len(wanted_words)
    model_settings = prepare_model_settings(SAMPLE_RATE,
                                            CLIP_DURATION_MS,
                                            WINDOW_SIZE_MS,
                                            WINDOW_SIZE_MS,
                                            DCT_COEFFICIENT_COUNT)
    input_shape = (1, model_settings['spectrogram_length'], model_settings['dct_coefficient_count'])

    def __init__(self, domain_key):
        self.maybe_download_and_extract_dataset(DATA_URL, ROOT_DIR, DATA_DIR)
        self.domain_key = domain_key
        self.prepare_data_index()
        self.prepare_background_data()
        self.prepare_processing_graph()
        self.X, self.y = self.fetch_all()


    def __getitem__(self, index):
        return self.X[index], int(self.y[index]), self.domain_key

    def __len__(self):
        return len(self.y)

    @staticmethod
    def maybe_download_and_extract_dataset(data_url, dest_directory, out_dir):
        """Download and extract data set tar file.
        Args:
          data_url: Web location of the tar file containing the data set.
          dest_directory: File path to extract data to.
        """
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                                 filepath)
                tf.logging.error('Please make sure you have enough free space and'
                                 ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                            statinfo.st_size)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            tarfile.open(filepath, 'r:gz').extractall(out_dir)

    def prepare_data_index(self):
        self.data = []
        for wanted_word in self.wanted_words:
            search_path = os.path.join(self.data_dir, wanted_word)
            wav_paths = glob.glob(search_path+'/*.wav')

            # choose samples corresponding to `domain_key`
            n_d = float(len(self.all_domain_key))
            idx = self.all_domain_key.index(self.domain_key)
            n_files = len(wav_paths)
            start = int((idx/n_d) * n_files)
            end = int(( (idx+1)/n_d ) * n_files)
            wav_paths = wav_paths[start:end]

            for wav_path in wav_paths:
                self.data.append({'label': wanted_word, 'file': wav_path})
        return

    def prepare_background_data(self):
        noise_name = self.domain_key[:-2] + '.wav'
        noise_file = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, noise_name)
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            self.noise_data = sess.run(wav_decoder,
                                       feed_dict={wav_filename_placeholder: noise_file}).audio.flatten()
        return

    def prepare_processing_graph(self):
        """Builds a TensorFlow graph to apply the input distortions.

        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.
        """
        desired_samples = self.model_settings['desired_samples']
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)

        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)

        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                     self.time_shift_offset_placeholder_,
                                     [desired_samples, -1])

        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                           [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
        background_mul = tf.multiply(self.background_data_placeholder_,
                                     self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=self.model_settings['window_size_samples'],
            stride=self.model_settings['window_stride_samples'],
            magnitude_squared=True)

        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=self.model_settings['dct_coefficient_count'])

    def fetch_all(self):
        sess = tf.InteractiveSession()
        time_shift = int((TIME_SHIFT_MS * SAMPLE_RATE) / 1000)
        background_volume_range = int(self.domain_key.split('_')[-1]) * 0.1

        sample_count = len(self.data)
        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, self.model_settings['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = self.model_settings['desired_samples']

        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for sample_index in xrange(0, sample_count):
            sample = self.data[sample_index]

            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset,
            }

            # Choose a section of background noise to mix in.
            background_samples = self.noise_data
            background_offset = np.random.randint(
                0, len(background_samples) - self.model_settings['desired_samples'])
            background_clipped = background_samples[background_offset:(
                    background_offset + desired_samples)]
            background_reshaped = background_clipped.reshape([desired_samples, 1])
            background_volume = np.random.uniform(background_volume_range - 0.1,
                                                  background_volume_range)
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            input_dict[self.foreground_volume_placeholder_] = 1

            # Run the graph to produce the output audio.
            data[sample_index, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
            label_index = self.word_to_index[sample['label']]
            labels[sample_index] = label_index

        input_frequency_size = self.model_settings['dct_coefficient_count']
        input_time_size = self.model_settings['spectrogram_length']
        X = data.reshape(data.shape[0], input_time_size, input_frequency_size, 1).transpose(0, 3, 1, 2)
        y = labels.astype(np.int8)
        return X, y


class Speech(DomainDatasetBase):
    """ Speech Command Datasets
    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleSpeech


class _SingleBiasedSpeech(_SingleSpeech):
    """ Biased version of _SingleSpeeh

        sample sizes of the combinations of domains and classes below are reduced to 50%
        | yes, no, up      | down, left, right | on, off, stop | go, cat, dog |
        | doing_the_dishes | dude_miaowing     | exercise_bike | running_tap  |
    """
    domain_class_pair = {'doing_the_dishes': ['yes', 'no', 'up'],
                         'dude_miaowing': ['down', 'left', 'right'],
                         'exercise_bike': ['on', 'off', 'stop'],
                         'running_tap': ['go', 'cat', 'dog']}

    def prepare_data_index(self):
        self.data = []
        for wanted_word in self.wanted_words:
            search_path = os.path.join(self.data_dir, wanted_word)
            wav_paths = glob.glob(search_path+'/*.wav')

            # choose samples corresponding to `domain_key`
            n_d = float(len(self.all_domain_key))
            idx = self.all_domain_key.index(self.domain_key)
            n_files = len(wav_paths)
            start = int((idx/n_d) * n_files)
            end = int(( (idx+1)/n_d ) * n_files)
            wav_paths = wav_paths[start:end]

            # make bias
            if wanted_word in self.domain_class_pair[self.domain_key[:-2]]:
                half = len(wav_paths) // 2
                wav_paths = wav_paths[:half]

            for wav_path in wav_paths:
                self.data.append({'label': wanted_word, 'file': wav_path})
        return


class BiasedSpeech(DomainDatasetBase):
    """ Biased version of Speech """
    SingleDataset = _SingleBiasedSpeech


if __name__ == "__main__":
    # dataset = Speech(['doing_the_dishes_1', 'doing_the_dishes_2',
    #                   'dude_miaowing_1', 'dude_miaowing_2'])
    dataset = _SingleBiasedSpeech('doing_the_dishes_1')
    from IPython import embed; embed()
