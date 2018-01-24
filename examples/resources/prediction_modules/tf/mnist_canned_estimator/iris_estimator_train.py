from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))

IRIS_TRAINING = "../../../data/tf/mnist_canned_estimator/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "../../../data/tf/mnist_canned_estimator/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Relative paths should be used. Refer Dos and Donts of Tensorflow
SAVED_MODEL = "../../../model_resources/tf/mnist_canned_estimator"
print(SAVED_MODEL)

if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'w') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, 'w') as f:
        f.write(raw)


training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)


test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


feature_columns = [tf.feature_column.numeric_column('x', shape=[4, 1])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir=SAVED_MODEL)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=2000)
