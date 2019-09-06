import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FNN():
    def __init__(self, X_size, Y_size, loss, optimizer, metrics):
        self.X_size = X_size
        self.Y_size = Y_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.GRAPH = self.graph()

    def graph(self):
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
        return model