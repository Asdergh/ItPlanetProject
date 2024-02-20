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
        for json_file in os.listdir(self.dataset_base_dir):
            
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

        self.chiters_activity_general_tensor = np.zeros(shape=(
            len(self.data_buffer[0]),
            len(self.data_buffer[0]["subject number: 1"]["steps"]["samples"]),
            len(self.data_buffer[0]["subject number: 1"]["steps"]["samples"][0]) + 3
        ))

        self.none_chiters_activity_general_tensor = np.zeros(shape=(
            len(self.data_buffer[1]),
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"]),
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 3
        ))

        self.samples_tensor = np.zeros(shape=(
            len(self.data_buffer[1]) * len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"]),
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 3
        ), dtype="float32")
        
        curent_samples_index = 0
        for (batch_number, data_batch) in enumerate(self.data_buffer[:-1]):

            
            if batch_number == 0:

                print("TEST ONE")
                for subject in data_batch:
                    
                    for sample in data_batch[subject]["steps"]["samples"]:
                        
                        distance_data = float(data_batch[subject]["steps"]["meters"])
                        steps_count = float(data_batch[subject]["steps"]["steps"])

                        sample_data = [float(feature) for feature in sample.values()]
                        sample_data[0] = sample_data[0] * (10 ** -5)
                        sample_data[1] = sample_data[1] / 120

                        sample_data.append(distance_data)
                        sample_data.append(steps_count)


                        print(sample_data[0], sample_data[1], type(sample_data[0]), type(sample_data[1]))

                        # last_vector = np.asarray([self.data_buffer[-1][curent_subject_id]["birth_date"], 
                        #                           self.data_buffer[-1][curent_subject_id]["weight"], 1])
                        
                        # self.chiters_activity_tensor[subject_number, sample_number, :-3] = sample_data
                        # self.chiters_activity_tensor[subject_number, sample_number, -3:] = last_vector
                        
                        self.samples_tensor[curent_samples_index, :-1] = sample_data
                        self.samples_tensor[curent_samples_index, -1] = 1

                        print(f"\nCurent samples number: [{curent_samples_index}], Samples: {self.samples_tensor[curent_samples_index]}\n")
                        print(sample_data)
                        curent_samples_index += 1
            
            elif batch_number == 1:
                
                print("TEST TWO")
                for subject in data_batch:
                    for sample in data_batch[subject]["steps"]["samples"]:
                        

                        distance_data = float(data_batch[subject]["steps"]["meters"])
                        steps_count = float(data_batch[subject]["steps"]["steps"])

                        sample_data = np.asarray([float(feature) for feature in sample.values()])
                        sample_data[0] = sample_data[0] * (10 ** -5)
                        sample_data[1] = sample_data[1] / 120
                        
                        sample_data.append(distance_data)
                        sample_data.append(steps_count)
                        
                        print(sample_data[0], sample_data[1], type(sample_data[0]), type(sample_data[1]))
                        # last_vector = np.asarray([data_batch[2][f"subject number: {subject_number}"]["birth_date"], 
                        #                           data_batch[2][subject_number]["weight"], 0])
                        
                        # self.none_chiters_activity_tensor[subject_number, sample_number, :-3] = sample_data
                        # self.none_chiters_activity_tensor[subject_number, sample_number, -3:] = last_vector
                        
                        self.samples_tensor[curent_samples_index, :-1] = sample_data
                        self.samples_tensor[curent_samples_index, -1] = 0

                        print(f"\nCurent samples number: [{curent_samples_index}], Samples: {self.samples_tensor[curent_samples_index]}\n")
                        curent_samples_index += 1

        
        return  self.samples_tensor

    # метод поректирование данных в результирующие тензоры для подачу на обучение нейронной сети
    def build_data(self):


        samples_features = self._make_data_tensors()
        
        # self.result_features = np.vstack((chiters_features, none_chiters_features))
        
        # self.shuffle_result_general_features = np.random.permutation(self.result_features)
        
        print(f"Samples features tensor: {samples_features}")
        self.shuffle_result_samples_features = np.random.permutation(samples_features)

        
        self.samples_train_data = self.shuffle_result_samples_features[:self.shuffle_result_samples_features.shape[0] // 2, :-1]
        self.samples_train_labels = self.shuffle_result_samples_features[:self.shuffle_result_samples_features.shape[0] // 2, -1]

        self.samples_test_data = self.shuffle_result_samples_features[self.shuffle_result_samples_features.shape[0] // 2:, :-1]
        self.samples_test_labels = self.shuffle_result_samples_features[self.shuffle_result_samples_features.shape[0] // 2:, -1]

        print(self.samples_train_data[0], self.samples_train_labels[0])
    









        
        

            


            

            