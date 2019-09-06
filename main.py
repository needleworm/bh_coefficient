"""
Byunghyun Ban
bhban@kaist.ac.kr
halfbottle@sangsang.farm
"""

import tensorflow as tf
import numpy as np
import graph
import os
import shutil
from tensorflow import keras
import time

# Flag Set
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', "train", "mode : train/ predict [default : train]")  # Mode
tf.flags.DEFINE_string('device', '/gpu:0', "device : /cpu:0 /gpu:0 [default : /gpu:0]")  # Device
tf.flags.DEFINE_bool('reset', "True", "reset : True/False")  # Reset train history
tf.flags.DEFINE_integer("num_X", "60", "SIZE. [Default : 60]") # size of input data
tf.flags.DEFINE_integer("num_Y", "17", "SIZE. [Default : 17]") # size of label
tf.flags.DEFINE_integer("training_batch_size", "128", "batch size for training. [default : 128]")
tf.flags.DEFINE_integer("test_batch_size", "128", "batch size for validation. [default : 128]")
tf.flags.DEFINE_integer("predict_batch_size", "128", "batch size for visualization. [default : 128]")
tf.flags.DEFINE_integer("num_epochs", "12", "how many epochs?. [default : 12]")

# Directory Setting
logs_dir = "logs"
test_x = "data/test_X.npy"
test_y = "data/test_Y.npy"
training_x = "data/training_X.npy"
training_y = "data/training_Y.npy"

# Hyperparameters
dropout_rate = 0.5
learning_rate = 1e-4

# Define Model
MODEL = graph.FNN

# Directory Reset
if FLAGS.mode is "predict":
    FLAGS.reset = False

if FLAGS.reset:
    print("** Note : Log Directory was Reset! **")
    if logs_dir in os.listdir():
        shutil.rmtree(logs_dir)
    os.mkdir(logs_dir)
    os.mkdir(logs_dir + "/train")
    os.mkdir(logs_dir + "/test")


# Training
def main():
    # Graph part
    print("Graph Initialization...")
    with tf.device(FLAGS.device):
        model = MODEL(X_size=FLAGS.num_X,
                        Y_size=FLAGS.num_Y,
                        loss="mean_squared_error",
                        optimizer="adam",
                        metrics=["accuracy"])

    #print(model.GRAPH.summary())

    # Summary Part
    tensorboard = keras.callbacks.TensorBoard(log_dir=logs_dir+"/{}".format(time.time()))

    # Data Reading
    print("** Reading Data...")
    Test_X = np.load(test_x)
    Test_Y = np.load(test_y)
    print("** Total " + str())
    if FLAGS.mode == "train":
        Train_X = np.load(training_x)
        Train_Y = np.load(training_y)

    # Training Part
    if FLAGS.mode == "train":
        model.GRAPH.fit(Train_X, Train_Y,
                    epochs=FLAGS.num_epochs,
                    batch_size=FLAGS.training_batch_size,
                    callbacks=[tensorboard])
        print("** Training Done **")
        print(model.GRAPH.summary())

    test_loss, test_acc = model.GRAPH.evaluate(Test_X, Test_Y)
    print("** test loss is : " + str(test_loss))
    print("** test acc is : " + str(test_acc))


main()