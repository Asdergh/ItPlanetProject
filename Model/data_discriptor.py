import numpy as np
import json as js
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA



class DataDiscriptor():


    def __init__(self) -> None:

            pass

    # функция загрузки данных из датасета включающего в себя набора .jsonl файлов
    # для подгрузки данных необходимо указать количество необходимых выборок пользователей
    # а так же файл с исходным расположением папки с .jsonl файлами
    # данная функция также ответсвенна за прогрузку .json файла с данными пользователя в случае если мы хотим протестировтаь работоспособность модели

    # variabel max_data: необходимое для процесса обучения количество выборок пользователей для обучения модели
    # type max_data: int

    # variabel start_position: индекс выборки пользователя с которой мы хотим начать подгрузку информации
    # type start_position: int

    # variabel base_dir: путь доя расплоложения дисректории с .jsonl файлами информации о пользователях
    # type base_dir: str

    # variabel base_file_path: путь до .json файла с информацией пользователя необходимой для теста нейронной сети
    # type base_file_path: str
    def _data_loader(self, max_data=None, start_position=0, base_dir=None, base_file_path=None):

        data_buffer = {}

        if base_dir is not None and (base_file_path is None or base_file_path is not None):

            if max_data is None:

                raise ValueError("[max_data is a reguared param] !!!")
            
            for json_file in os.listdir(base_dir)[:-1]:

                print(json_file)
                curent_file = os.path.join(base_dir, json_file)
                curent_data_buffer = {}
                with open (curent_file) as file:

                    json_data = file.readlines()
                    json_data = json_data[start_position: max_data]

                    for (json_number, json_per_line) in enumerate(json_data):
                        
                    
                        json_format = js.loads(json_per_line)
                        curent_data_buffer[f"subject_number: {json_number}"] = json_format
                
                data_buffer[json_file] = curent_data_buffer


            person_id_list = []
            for data_batch in data_buffer.keys():
                for subject in data_buffer[data_batch].keys():

                    person_id_list.append(data_buffer[data_batch][subject]["profile_id"])


            person_log_file = os.path.join(base_dir, os.listdir(base_dir)[-1])
            person_info_buffer = {}

            with open(person_log_file) as file:

                json_data = file.readlines()
                for (json_number, json_per_line) in enumerate(json_data):

                    json_format_data = js.loads(json_per_line)
                    if json_format_data["id"] in person_id_list:

                        person_info_buffer[json_format_data["id"]] = json_format_data
            
            return data_buffer, person_info_buffer


        elif base_dir is None and base_file_path is not None:

            with open(base_file_path, "r") as json_file:

                data_buffer = js.load(json_file)
                return data_buffer

        elif base_dir is None and base_file_path is not None:

            raise ValueError("[both datasets paths are none] !!!")



    # данная функция используется для формирования тензора с параметрами пользователя необходимого для тестирования нейронной сети
    
    # variabel data_buffer: словарь со всеми данными о пользователе сформированный функцией self._data_load(base_file_path!=None) для тестирования модели
    # type data_buffer: python dict
    def _generate_input_samples(self, data_buffer):

        profile = data_buffer["profile"]
        sessions = data_buffer["sessions"]

        result_data = []
        profile_info = [float(profile[feature]) for feature in profile.keys() if feature not in ["sex", "id", "personal_goals"]]
        
        for session in sessions:
            
            min_variation = 1000000000000
            session_samples = np.asarray([[float(sample[feature]) for feature in sample.keys() if feature != "duration"] for sample in session["steps"]["samples"]])
            mean_vector = np.asarray([np.mean(session_samples[:, vector_number]) for vector_number in range(session_samples.shape[1])])

            optim_vector = np.zeros(shape=mean_vector.shape)

            session_general_data = [float(session[feature]) for feature in session.keys() if feature not in ["steps", "id", "timezone", "profile_id"]]
            session_general_data[0] /= 360

            session_steps_data = [float(session["steps"][feature]) for feature in session["steps"].keys() if feature != "samples"]
            session_steps_data[0] *= 3.6e+6
            session_steps_data[1] *= 3.6e+6

            for curent_vector in session_samples:

            
                if min_variation > np.dot(curent_vector, mean_vector):
                    
                    min_variation = np.dot(curent_vector, mean_vector)
                    optim_vector = curent_vector
            
            optim_vector = list(optim_vector)
            optim_vector[0] *= 3.6e+6

            optim_vector += (session_general_data + session_steps_data + profile_info)
            result_data.append(optim_vector)
        
        result_data = np.asarray(result_data)
        return result_data
    
    
    def _generate_data(self, data_buffer, person_info_buffer):

        data_list = [] 
        subjects_features = ["start_millis", "stop_millis", "skllzz", 
                                "activity_day", "skllzz_with_artifacts", "skllzz_without_artifacts", 
                                "steps", "day", "meters", 
                                "birth_date", "hr_rest", "hr_max", 
                                "weight", "kkal"]
        
        for (batch_number, data_batch) in enumerate(data_buffer):

            print(data_batch)
            sub_info = []
            subjects = [key for key in data_buffer[data_batch].keys()]
            subjects_id = [data_buffer[data_batch][subject]["profile_id"] for subject in subjects]

            for feature in subjects_features:
                
                subject_feature_list = []
                for (subject, subject_id) in zip(subjects, subjects_id):
                    
                    if feature in ["day", "meters", "steps"]:

                        if feature in data_buffer[data_batch][subject]["steps"].keys():
                            subject_feature_list.append(float(data_buffer[data_batch][subject]["steps"][feature]))

                        else:
                            subject_feature_list.append(0.0)
                    
                    elif feature in ["hr_max", "hr_rest", "weight", "birth_date"]:

                        if feature in person_info_buffer[subject_id].keys():
                            subject_feature_list.append(float(person_info_buffer[subject_id][feature]))
                        
                        else:
                            subject_feature_list.append(0.0)
                    
                    elif feature in ["stop_millis", "start_millis"]:

                        if feature in data_buffer[data_batch][subject].keys():
                            
                            formated_millis = float(data_buffer[data_batch][subject][feature]) * 3.6e+6
                            subject_feature_list.append(formated_millis)

                        else:
                            subject_feature_list.append(0.0)

                    else:

                        if feature in data_buffer[data_batch][subject].keys():
                            subject_feature_list.append(float(data_buffer[data_batch][subject][feature]))
                        
                        else:
                            subject_feature_list.append(0.0)
                
                sub_info.append(subject_feature_list)
            sub_info_tensor = np.asarray(sub_info, dtype="float32")
            


            for (subject_number, subject) in enumerate(data_buffer[data_batch].keys()):

                add_vector = list(sub_info_tensor[:, subject_number]) + [batch_number, ]

                if "samples" in data_buffer[data_batch][subject]["steps"].keys():

                    min_var = 1000000000000000000000000000
                    samples_data = np.asarray([[float(sample[feature]) for feature in sample.keys() if feature != "duration"] for sample in data_buffer[data_batch][subject]["steps"]["samples"]])
                    mean_vector = [np.mean(samples_data[:, vector_number]) for vector_number in range(samples_data.shape[1])]
                    optim_vector = None

                    for sample in data_buffer[data_batch][subject]["steps"]["samples"]:
                        
                        sample_vector = np.asarray([float(sample[feature]) for feature in sample.keys() if feature != "duration"])
                        if min_var > (np.dot(sample_vector, mean_vector)):

                            min_var = (np.dot(sample_vector, mean_vector))
                            optim_vector = list(sample_vector)


                    optim_vector += add_vector
                    optim_vector = np.asarray(optim_vector)
                    optim_vector[0] *= 3.6e+6


                else:
                    optim_vector = [0.0, 0.0] + add_vector
                
                data_list.append(optim_vector)



        all_features = ["stamp_millis", "steps"] + subjects_features + ["class_labels", ]
        data_tensor = np.asarray(data_list)
        data_frame = pd.DataFrame(data=data_tensor,
                                columns=all_features)
        
        return (data_frame, data_tensor)
    

    def _expand_randomization(self, data_tensor, need_shape):

    
        permutated_data_tensor = np.random.permutation(data_tensor)
        random_normal_distrib = np.random.normal(0.19, 1.926, ((permutated_data_tensor.shape[1] - 1), need_shape))
        result_expand = np.dot(permutated_data_tensor[:, :-1], random_normal_distrib)
        result_expand_std = (result_expand - np.mean(result_expand)) / np.std(result_expand)

        result_tensor = np.zeros(shape=(result_expand.shape[0], result_expand.shape[1] + 1))
        result_tensor[:, :-1] = result_expand_std
        result_tensor[:, -1] = permutated_data_tensor[:, -1]

        return result_tensor

    def generate_data(self, base_dir):

        data_buffer, person_info_buffer = self._data_loader(max_data=2000, base_dir=base_dir)
        data_frame, data_tensor= self._generate_data(data_buffer=data_buffer, person_info_buffer=person_info_buffer)
        print(data_tensor)
        expanded_data_tensor = self._expand_randomization(data_tensor=data_tensor, need_shape=800)

        train_data_tensor = expanded_data_tensor[:expanded_data_tensor.shape[0] // 2, :-1]
        train_data_labels = expanded_data_tensor[:expanded_data_tensor.shape[0] // 2, -1]

        validation_data_tensor = expanded_data_tensor[expanded_data_tensor.shape[0] // 2:, :-1]
        validation_data_labels = expanded_data_tensor[expanded_data_tensor.shape[0] // 2:, -1]

        return (train_data_tensor, train_data_labels), (validation_data_tensor, validation_data_labels)

        


    



if __name__ == "__main__":

    data_worker = DataDiscriptor()
    (train_data, train_labels), (validation_data, validation_labels) = data_worker.generate_data(base_dir="c:\\Users\\1\\Desktop\\datasets")

    plt.style.use("dark_background")
    fig, axis = plt.subplots(nrows=2)
    axis[0].imshow(train_data[:100, :300], cmap="magma")
    axis[1].imshow(validation_data[:100, :300], cmap="viridis")

    axis[0].grid()
    axis[1].grid()

    axis[0].set_title("train data")
    axis[1].set_title("validation data")
    plt.show()

    








        
        

            


            

            