import numpy as np
import json as js
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



class DataDiscriptor():


    def __init__(self) -> None:

            pass

    
    def _data_loader(self, start_position, max_data, base_dir):
          
        data_buffer = {}

        for json_file in os.listdir(base_dir)[:-1]:


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



    def _generate_data(self, person_info_buffer, data_buffer, generation_type="all"):




            data_list = []
            for (batch_number, data_batch) in enumerate(data_buffer.keys()):

                gender_labels = {
                    "male": 1.0,
                    "female": 2.0
                }

                personal_goal_labels = {
                    "unknown_goal": 0.0,
                    "lose_weight": 1.0,
                    "childrens_training": 2.0,
                    "relief": 3.0,
                    "strength": 4.0,
                    "antistress": 5.0,
                    "learn_swim": 6.0,
                    "flexibility": 7.0,
                    "body_balance": 8.0,
                    "fun": 9.0,
                    "rehabilitation": 10.0
                }

                
                for subject in data_buffer[data_batch].keys():


                    curent_person_id = data_buffer[data_batch][subject]["profile_id"]
                    curent_person_object = person_info_buffer[curent_person_id]
                
                    
                    if "samples" in data_buffer[data_batch][subject]["steps"]:
                        
                        
                        if "skllzz_with_artifacts" in data_buffer[data_batch][subject].keys():

                            subject_general_data = [
                                (float(data_buffer[data_batch][subject]["stop_millis"]) * (10 ** -5)) - (float(data_buffer[data_batch][subject]["start_millis"]) * (10 ** -5)),
                                float(data_buffer[data_batch][subject]["skllzz"]),
                                float(data_buffer[data_batch][subject]["activity_day"]),
                                float(data_buffer[data_batch][subject]["skllzz_with_artifacts"]),
                                float(data_buffer[data_batch][subject]["skllzz_without_artifacts"]), 
                            ]

                        else:

                            subject_general_data = [
                                (float(data_buffer[data_batch][subject]["stop_millis"]) * (10 ** -5)) - (float(data_buffer[data_batch][subject]["start_millis"]) * (10 ** -5)),
                                float(data_buffer[data_batch][subject]["skllzz"]),
                                float(data_buffer[data_batch][subject]["activity_day"]),
                                0.0,
                                float(data_buffer[data_batch][subject]["skllzz_without_artifacts"]), 
                            ]
                        
                        subject_pysical_data = [
                            float(data_buffer[data_batch][subject]["steps"]["steps"]),
                            float(data_buffer[data_batch][subject]["steps"]["day"]),
                            float(data_buffer[data_batch][subject]["steps"]["meters"]) / 100
                        ]

                        subject_personal_data = [
                            float(curent_person_object["birth_date"]) / 360,
                            float(curent_person_object["hr_rest"]),
                            float(curent_person_object["hr_max"]),
                            float(curent_person_object["weight"])
                        ]

                        if "personal_goals" in curent_person_object.keys():

                            subject_personal_goals = [personal_goal_labels[goal] for goal in curent_person_object["personal_goals"]]
                            if len(subject_personal_goals) != len(personal_goal_labels):

                                kernel = [personal_goal_labels["unknown_goal"] for _ in range(len(personal_goal_labels) - len(subject_personal_goals))]
                                subject_personal_goals += kernel
                        
                        else:

                            subject_personal_goals = [personal_goal_labels["unknown_goal"] for _ in range(len(personal_goal_labels))]
                        
                    

                        if "skllzz_with_artifacts" in data_buffer[data_batch][subject].keys():
                            subject_general_data.append(data_buffer[data_batch][subject]["skllzz_with_artifacts"])
                        
                        else:
                            subject_general_data.append(0.0)
                        
                        add_vector = subject_general_data + subject_pysical_data + subject_personal_data + subject_personal_goals
                        add_vector.append(float(batch_number))
                        




                        if generation_type == "per_day":

                            for sample in data_buffer[data_batch][subject]["steps"]["samples"]:

                                sample_data = [float(feature) for feature in sample.values()]
                                sample_data[0] *= (10 ** -5)
                                sample_data[1] /= 3600.0

                                sample_data += add_vector
                                
                                # if len(sample_data) == 18:

                                #     data_list.append(sample_data)
                                
                                data_list.append(sample_data)
                        
                        elif generation_type == "all":

                            samples_data_list = []
                            for sample in data_buffer[data_batch][subject]["steps"]["samples"]:
                                
                                sample_data = [float(feature) for feature in sample.values()]
                                sample_data[0] *= (10 ** -5)
                                sample_data[1] /= 3600.0
                                
                                samples_data_list += sample_data
                                samples_data_list += add_vector

                                data_list.append(samples_data_list)

                        else:

                            raise ValueError("you must condider data generation type ['all' or 'per_daya']")
                        
            return data_list
    
    def _formulate_data(self, data_list, generation_type, max_from_samples=None):


        if generation_type == "all":

                if max_from_samples is None:

                    max_sample_len = max([len(sample) for sample in data_list])

                else:

                    max_sample_len = max_from_samples

                    data_tensor = np.zeros(shape=(len(data_list), max_sample_len))

                    for sample_number in range(data_tensor.shape[0]):

                        if data_list[sample_number] != data_tensor.shape[1]:

                            class_label = data_list[sample_number][-1]
                            none_class_list = data_list[sample_number][:-1]
                            kernel_add = [0. for _ in range((max_sample_len - len(none_class_list)) - 1)]
                            
                            result_vector = none_class_list + kernel_add
                            result_vector.append(class_label)
                            result_vector = np.asarray(result_vector)
                        
                        else:

                            result_vector = np.asarray(data_list[sample_number])
                        
                        data_tensor[sample_number] = result_vector
            

        elif generation_type == "per_day":
                
            data_tensor = np.asarray(data_list)
        
        return data_tensor
    

    def generate_data(self, base_dir, max_data_position, generation_type="all", 
                      max_from_samples=None, start_data_position=0):

        pca_estimator = PCA(n_components=4)

        data_buffer, person_info_buffer = self._data_loader(base_dir=base_dir, max_data=max_data_position, start_position=start_data_position)
        data_list = self._generate_data(data_buffer=data_buffer, person_info_buffer=person_info_buffer, generation_type=generation_type)
        data_tensor = self._formulate_data(data_list=data_list, generation_type=generation_type)

        data_tensor_std = (data_tensor - np.mean(data_tensor)) / np.std(data_tensor)
        data_tensor_PCA = pca_estimator.fit_transform(data_tensor_std)
        permutated_data_tensor_PCA = np.random.permutation(data_tensor_PCA)

        train_data = permutated_data_tensor_PCA[: permutated_data_tensor_PCA.shape[0] // 2, :-1]
        test_data = permutated_data_tensor_PCA[permutated_data_tensor_PCA.shape[0] // 2: , :-1]

        train_labels = permutated_data_tensor_PCA[: permutated_data_tensor_PCA.shape[0] // 2, -1]
        test_labels = permutated_data_tensor_PCA[permutated_data_tensor_PCA.shape[0] // 2: , -1]

        
        return (train_data, test_data), (train_labels, test_labels)

    



if __name__ == "__main__":

    data_worker = DataDiscriptor()
    (train_data, test_data), (train_labels, test_labels) = data_worker.generate_data(base_dir="c:\\Users\\1\\Desktop\\datasets", max_data_position=2000, generation_type="per_day")


    print(train_data, train_data.shape)
    








        
        

            


            

            