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
            len(self.data_buffer[1]["subject number: 1"]["steps"]["samples"][0]) + 4
        ), dtype="float32")
        

        curent_samples_index = 0
        for (batch_number, data_batch) in self.data_buffer[:-1]:
            
            for (subject_number, subject) in enumerate(data_batch):


                subject_data = np.asarray([[float(feature)
                                        for feature in sample.value()] 
                                        for sample in data_batch[subject]["steps"]["samples"]])
                
                add_data = [float(data_batch[subject]["steps"]["steps"]),
                            float(data_batch[subject]["steps"]["meters"]),
                            float(data_batch[subject]["steps"]["day"])]
                    
                if batch_number == 0:

                    add_data.append(0)
                    
                else:

                    add_data.append(1)
                    

                self.samples_tensor[subject_number * subject_data.shape[0]: (subject_number + 1) * subject_data.shape[0], :-2] = subject_data
                self.samples_tensor[subject_number * subject_data.shape[0]: (subject_number + 1) * subject_data.shape[0], -2:] = add_data
                curent_samples_index += 1
        
        
        print(f"Final result: {self.samples_tensor}")
        return self.samples_tensor


                
                



        
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
    









        
        

            


            

            