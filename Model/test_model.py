#!/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from data_discriptor import DataDiscriptor


class Model(DataDiscriptor):

    def __init__(self) -> None:
        super().__init__()

    def _generate_model(self, need_shape):

        input_tensor = tf.keras.Input(shape=(need_shape[1], ))

        self.layer = tf.keras.layers.Dense(100,
                                           input_shape=(need_shape[0], ),
                                           activation="linear",
                                           activity_regularizer=tf.keras.regularizers.L1(
                                               0.001),
                                           kernel_regularizer=tf.keras.regularizers.L2(0.001))(input_tensor)

        self.layer = tf.keras.layers.Dense(10,
                                           activation="linear",
                                           activity_regularizer=tf.keras.regularizers.L1(
                                               0.01),
                                           kernel_regularizer=tf.keras.regularizers.L2(0.01))(self.layer)

        #layer = tf.keras.layers.Dropout(0.5)(layer)
        self.last_layer = tf.keras.layers.Dense(
            1, activation="sigmoid")(self.layer)
        self.model = tf.keras.Model(input_tensor, self.last_layer)
        self.model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def _fit_model(self, train_data, train_label, validation_data=None, validation_label=None):

        print(train_data.shape, train_label.shape)
        print(validation_data.shape, validation_label.shape)
        self.model_history = self.model.fit(train_data, train_label,
                                            epochs=100,
                                            batch_size=30,
                                            validation_data=(validation_data, validation_label))

        self.model.save(
            "C:\\Users\\1\\Desktop\\ItPLanetProject2\\SavedModels\\first_model.keras")

    def _show_history(self):

        model_stats = [[np.asarray(self.self.model_history.history["loss"]), np.asarray(self.model_history.history["val_loss"])],
                       [np.asarray(self.model_history.history["accuracy"]), np.asarray(self.model_history.history["val_accuracy"])]]

        labels = [["loss", "val_loss"], ["accuracy", "val_accuracy"]]
        colors = [["red", "orange"], ["green", "blue"]]

        plt.style.use("dark_background")
        fig, axis = plt.subplots(nrows=2, ncols=2)
        for sample in range(len(axis)):

            axis[sample][0].plot(range(1, model_stats[sample][0].shape[0] + 1),
                                 model_stats[sample][0], color=colors[sample][0], label=labels[sample][0])
            axis[sample][0].fill_between(range(1, model_stats[sample][0].shape[0] + 1), model_stats[sample]
                                         [0] - 0.12, model_stats[sample][0] + 0.12, color=colors[sample][0], alpha=0.25)

            axis[sample][1].plot(range(1, model_stats[sample][1].shape[0] + 1),
                                 model_stats[sample][1], color=colors[sample][1], label=labels[sample][1])
            axis[sample][1].fill_between(range(1, model_stats[sample][1].shape[0] + 1), model_stats[sample]
                                         [1] - 0.12, model_stats[sample][1] + 0.12, color=colors[sample][1], alpha=0.25)

            axis[sample][0].legend(loc="lower right")
            axis[sample][0].grid()

            axis[sample][1].legend(loc="lower right")
            axis[sample][1].grid()

        plt.show()

    def train_model(self):

        (train_data, train_labels), (validation_data, validation_labels) = self.generate_data(
            base_dir="c:\\Users\\1\\Desktop\\datasets")

        self._generate_data()
        self._fit_model(train_data=train_data, train_labels=train_labels,
                        validation_data=validation_data, validation_labels=validation_labels)
        self._show_history()

    def make_predictions(self, subject_info_file_dir):

        model = tf.keras.saving.load_model(
            "C:\\Users\\1\\Desktop\\ItPLanetProject2\\SavedModels\\first_model.keras")
        testing_data = self.generate_data_testing(subject_info_file_dir)
        predictions = model.predict(testing_data[:, :-1])

        predictions_tensor = [prediction[0] for prediction in predictions]
        result_prediction_tensor = np.asarray(
            [sum(predictions_tensor) / len(predictions_tensor), ] + predictions_tensor)

        return result_prediction_tensor

        
if __name__ == "__main__":

    model_object = Model()
    prediction = model_object.make_predictions(
        subject_info_file_dir="C:\\Users\\1\\Desktop\\ItPLanetProject2\\input-sample.json")
    print(prediction)
