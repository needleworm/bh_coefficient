import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FNN():
    def __init__(self, X_size, Y_size, loss, optimizer, metrics, reset, logdir):
        self.X_size = X_size
        self.Y_size = Y_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.GRAPH = self.graph(reset, logdir)

    def graph(self, reset, logdir):
        if reset:
            model = tf.keras.Sequential()
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(self.Y_size, activation="relu"))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)
        else:
            model = keras.models.load_model(logdir+"/saved_model.h5")
        return model
