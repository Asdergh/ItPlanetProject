import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_discriptor import DataDiscriptor





class Model():

    def __init__(self, first_model_depth, second_model_depth, L1_regularization_rate, L2_regularization_rate) -> None:
        

        self.first_model_depth = first_model_depth
        self.second_model_depth = second_model_depth
        self.L1_reg_rate = L1_regularization_rate
        self.L2_reg_rate = L2_regularization_rate

        self.data_discription = DataDiscriptor(dataset_base_dir="C:\\Users\\1\\Desktop\\datasets")
        self.data_discription.load_data()
        self.data_discription.build_data()

        self.first_model_constant_width = len(self.data_discription.data_buffer[0]["subject number: 1"]["steps"]["samples"][0])
        self.second_model_constant_width = len(self.data_discription.data_buffer[0]["subject number: 1"]["steps"]["samples"])

        


    # метод для генерации модели глубокого обучение
    # данный метод создает модель по тиму бинарного классификатра
        
    def build_model(self):

        
        self.first_input_tensor = tf.keras.Input(shape=(self.first_model_constant_width, ))
        self.second_input_tensor = tf.keras.Input(shape=(self.second_model_constant_width, ))
        


        self.first_model_dense_layer = tf.keras.layers.Dense(self.first_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_input_tensor)
        
        self.first_model_dense_layer = tf.keras.layers.Dense(self.first_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_model_dense_layer)
        
        self.first_model_dense_layer = tf.keras.layers.Dense(self.first_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_model_dense_layer)
        
        self.first_model_dense_layer = tf.keras.layers.Dense(self.first_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_model_dense_layer)
        
        self.first_model_dense_layer = tf.keras.layers.Dense(self.first_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_model_dense_layer)
        
        self.first_model_last_layer = self.first_model_dense_layer = tf.keras.layers.Dense(1, activation="sigmoid",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.first_model_dense_layer)
        


        self.second_model_dense_layer = tf.keras.layers.Dense(self.second_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_input_tensor)
        
        self.second_model_dense_layer = tf.keras.layers.Dense(self.second_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_model_dense_layer)
        
        self.second_model_dense_layer = tf.keras.layers.Dense(self.second_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_model_dense_layer)
        
        self.second_model_dense_layer = tf.keras.layers.Dense(self.second_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_model_dense_layer)
        
        self.second_model_dense_layer = tf.keras.layers.Dense(self.second_model_constant_width, activation="relu",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_model_dense_layer)
        
        self.second_model_last_layer = self.first_model_dense_layer = tf.keras.layers.Dense(1, activation="sigmoid",
                                                             kernel_regularizer=tf.keras.regularizers.L1(self.L1_reg_rate),
                                                             activity_regularizer=tf.keras.regularizers.L2(self.L2_reg_rate))(self.second_model_dense_layer)
        

        self.first_model = tf.keras.Model(self.first_input_tensor, self.first_model_last_layer)
        self.second_model = tf.keras.Model(self.second_input_tensor, self.second_model_last_layer)



        self.first_model.compile(
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.01),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=tf.metrics.Accuracy()
        )

        self.second_model.compile(
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.01),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=tf.metrics.Accuracy()
        )
    
    # метод для подгрузки данных в модель нейронной сети
        
    def fit_model(self):

        self.first_model_history = self.first_model.fit(
            self.data_discription.samples_train_data,
            self.data_discription.samples_train_labels,
            batch_size=30,
            epochs=100
        )

        self.second_model_data = np.zeros(shape=(
            len(self.data_discription[0]),
            len(self.data_discription[0]["subject number: 1"]["steps"]["samples"]) + 1
        ))
        
        curent_subject_number = 0
        for subject in self.data_discription[0]:

            curent_samples = subject["steps"]["samples"]
            curent_samples_vector = []
            for sample in curent_samples:
                
                sample_tensor = np.asarray(sample.values())
                sample_tensor = np.expand_dims(sample_tensor, axis=0)
                prediction = self.first_model.predict(sample_tensor)

                curent_samples_vector.append(prediction[0])
            
            curent_samples_vector.append(0)
            curent_samples_vector = np.asarray(curent_samples_vector)
            self.second_model_data[curent_subject_number] = curent_samples_vector

            curent_subject_number += 1
        
        for subject in self.data_discription[1]:

            curent_samples = subject["steps"]["samples"]
            curent_samples_vector = []
            for sample in curent_samples:
                
                sample_tensor = np.asarray(sample.values())
                sample_tensor = np.expand_dims(sample_tensor, axis=0)
                prediction = self.first_model.predict(sample_tensor)

                curent_samples_vector.append(prediction[0])
            
            curent_samples_vector.append(0)
            curent_samples_vector = np.asarray(curent_samples_vector)
            self.second_model_data[curent_subject_number] = curent_samples_vector

            curent_subject_number += 1
        
        self.second_train_data = self.second_model_data[: self.second_model.shape[0] // 2, :-1]
        self.secon_train_labels = self.second_model_data[: self.second_model.shape[0] // 2 , -1]

        self.second_test_data = self.second_model_data[self.second_model.shape[0] // 2: , :-1]
        self.second_test_labels = self.second_model_data[self.second_model.shape[0] // 2, -1]


        self.second_model_history = self.second_model.fit(
            self.second_train_data,
            self.second_train_labels,
            validation_data=(self.second_test_data, self.second_test_labels),
            batch_size=30,
            epochs=100

        )
    
    
    # метод для демонстрации хода обучения
    # выводиться информация об метрике обучения и потерях подсчитанных функцией потерь
    def show_history(self):

        plt.style.use("dark_background")
        fig, axis = plt.subplots(nrows=3)

        models_loss = [np.asarray(self.first_model_history.history["loss"]), 
                       np.asarray(self.second_model_history.history["loss"]),
                       np.asarray(self.second_model_history.history["val_loss"])]
        
        models_acc = [np.asarray(self.first_model_history.history["acc"]),
                      np.asarray(self.second_model_history.history["acc"]),
                      np.asarray(self.second_model_history.history["val_acc"])]
        
        colors = ["green", "blue", "orange"]
        labels = [["first_model loss", "first_model acc"], ["second_model loss", "second_model acc"], ["second_model val_loss", "second_model val_acc"]]
        linestyles = ["", "", "--"]

        for (samples_number, samples) in enumerate(zip(models_loss, models_acc)):

            axis[samples_number].plot(range(1, samples[0].shape[0] + 1), samples[0], color=colors[samples_number], label=labels[samples_number], linestyle=linestyles[samples_number])
            axis[samples_number].legend(loc="upper left")
        
        plt.show()


if __name__ == "__main__":

    model = Model(
        first_model_depth=5,
        second_model_depth=5,
        L1_regularization_rate=0.1,
        L2_regularization_rate=0.01
    )

    model.build_model()
    model.fit_model()
    model.show_history() 

    

        







            


            

        
        
    