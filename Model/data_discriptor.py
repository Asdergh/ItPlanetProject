import numpy as np
import json as js
import os
import matplotlib.pyplot as plt



class DataDiscriptor():

    def __init__(self, dataset_base_dir) -> None:
        

        self.dataset_base_dir = dataset_base_dir

    # метод подгрузки данных из файлов (данный метод специализирован на получения данных оного формата .jsonl)
    def load_data(self):

        self.data_buffer = []
        print(os.listdir(self.dataset_base_dir)[:-1])
        for json_file in os.listdir(self.dataset_base_dir)[:-1]:
            
            print(json_file)
            curent_dir = os.path.join(self.dataset_base_dir, json_file)
            curent_data_buffer = {}
            with open(curent_dir, "r") as file:

                data = file.readlines()
                for (json_number, json_per_line) in enumerate(data):

                    json_format_data = js.loads(json_per_line)
                    curent_data_buffer[f"subject number: {json_number}"] = json_format_data

            self.data_buffer.append(curent_data_buffer)


    # TODO перепичать алгоритм получения тензоров выборок под новый метод обучения
    #   1) убрать общие тензоры активности пользователей (как читеров так и честных пользователей)
    def _make_data_tensors(self):

        # self.chiters_activity_general_tensor = np.zeros(shape=(
        #     len(self.data_buffer[0]),
        #     len(self.data_buffer[0]["subject number: 1"]["steps"]["samples"]),
        #     len(self.data_buffer[0]["subject number: 1"]["steps"]["samples"][0]) + 3
        # ))

        # self.none_chiters_activity_general_tensor = np.zeros(shape=(
        #     len(self.data_buffer[1]),
        #     len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"]),
        #     len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 3
        # ))

        self.samples_tensor = np.zeros(shape=(
            len(self.data_buffer[1]) * len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"]),
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 4
        ), dtype="float32")
        

        class_marks = [0, 1]
        for (batch_number, data_batch) in enumerate(self.data_buffer):
            
            print(class_marks[batch_number])
            for (subject_number, subject) in enumerate(data_batch):

                subject_data = np.asarray([[float(feature)
                                            for feature in sample.values()] 
                                            for sample in data_batch[subject]["steps"]["samples"]])
                
                subject_data[:, 0] += (10 ** -5)
                subject_data[:, 1] /= 120

                add_data = [float(data_batch[subject]["steps"]["steps"]),
                            float(data_batch[subject]["steps"]["meters"]),
                            float(data_batch[subject]["steps"]["day"]), class_marks[batch_number]]
                
                add_data = np.asarray(add_data)

                
                self.samples_tensor[subject_number * subject_data.shape[0]: (subject_number + 1) * subject_data.shape[0], :-4] = subject_data
                self.samples_tensor[subject_number * subject_data.shape[0]: (subject_number + 1) * subject_data.shape[0], -4:] = add_data
        
        
        print(f"Final result: {self.samples_tensor}, And its shape: {self.samples_tensor.shape}")
        return self.samples_tensor



    # метод поректирование данных в результирующие тензоры для подачу на обучение нейронной сети
    def build_data(self):


        samples_features = self._make_data_tensors()
        
        # self.result_features = np.vstack((chiters_features, none_chiters_features))
        
        # self.shuffle_result_general_features = np.random.permutation(self.result_features)
        
        print(f"Samples features tensor: {samples_features}")
        
        self.samples_train_data = samples_features[ :samples_features.shape[0] // 2, :-1 ]
        self.samples_train_labels = samples_features[ :samples_features.shape[0] // 2 , -1 ]

        print(self.samples_train_data, self.samples_train_labels)








        
        

            


            

            