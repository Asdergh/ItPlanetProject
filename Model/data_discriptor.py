import numpy as np
import json as js
import os
import matplotlib.pyplot as plt



class DataDiscriptor():

    def __init__(self, dataset_base_dir) -> None:
        

        self.dataset_base_dir = dataset_base_dir

    # метод подгрузки данных из файлов (данный метод специализирован на получения данных оного формата .jsonl)
    def _load_data(self):

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
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 1
        ))
        
        curent_samples_index = 0
        for (batch_number, data_batch) in enumerate(self.data_buffer[:-2]):

            
            if batch_number == 0:
                
                for (subject_number, subject) in enumerate(data_batch):
                    for (sample_number, sample) in enumerate(subject["steps"]["samples"]):
                        
                        sample_data = np.asarray([feature for feature in sample.values()])
                        last_vector = np.asarray([data_batch[2][f"subject number: {subject_number}"]["birth_date"], 
                                                  data_batch[2][subject_number]["weight"], 1])
                        
                        self.chiters_activity_tensor[subject_number, sample_number, :-3] = sample_data
                        self.chiters_activity_tensor[subject_number, sample_number, -3:] = last_vector
                        
                        self.samples_tensor[curent_samples_index, :-1] = sample_data
                        self.samples_tensor[curent_samples_index, :-1] = 1

                        curent_samples_index += 1
            
            elif batch_number == 1:

                for (subject_number, subject) in enumerate(data_batch):
                    for (sample_number, sample) in enumerate(subject["steps"]["samples"]):
                        
                        sample_data = np.asarray([feature for feature in sample.values()])
                        last_vector = np.asarray([data_batch[2][f"subject number: {subject_number}"]["birth_date"], 
                                                  data_batch[2][subject_number]["weight"], 0])
                        
                        self.none_chiters_activity_tensor[subject_number, sample_number, :-3] = sample_data
                        self.none_chiters_activity_tensor[subject_number, sample_number, -3:] = last_vector

                        self.samples_tensor[curent_samples_index, :-1] = sample_data
                        self.samples_tensor[curent_samples_index, -1] = 0

                        curent_samples_index += 1

        
        return self.chiters_activity_tensor, self.none_chiters_activity_tensor, self.samples_tensor

    # метод поректирование данных в результирующие тензоры для подачу на обучение нейронной сети
    def bulid_data(self):


        chiters_features, none_chiters_features, samples_features = self._make_data_tensors()
        
        self.result_features = np.vstack((chiters_features, none_chiters_features))
        
        self.shuffle_result_general_features = np.random.permutation(self.result_features)
        self.shuffle_result_samples_features = np.random.permutation(self.samples_features)

        self.general_train_data = self.shuffle_result_general_features[:self.shuffle_result_general_features.shape[0] // 2, :, :-1]
        self.general_train_labels = self.shuffle_result_general_features[:self.shuffle_result_general_features.shape[0] // 2, :, -1]
        
        self.samples_train_data = self.shuffle_result_samples_features[:self.shuffle_result_samples_features.shape[0] // 2, :-1]
        self.samples_train_labels = self.shuffle_result_samples_features[:self.shuffle_result_samples_features.shape[0] // 2, -1]

        self.samples_test_data = self.shuffle_result_samples_features[self.shuffle_Result_samples_features.shape[0] // 2:, :-1]
        self.samples_test_labels = self.shuffle_result_samples_features[self.shuffle_Result_samples_features.shape[0] // 2:, -1]
    









        
        

            


            

            