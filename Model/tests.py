import json as js
import numpy as np
import os
import random as rd


# datasets = []
# base_dir = "C:\\Users\\1\\Desktop\\SPECIAL_FOR_IT_PLANET\\datasets"
# for json_file in os.listdir(base_dir):
    
#     curent_file = os.path.join(base_dir, json_file)
#     with open(curent_file, "r") as file:

#         data = file.readlines()
#         sub_dataset = {}

#         for (json_number, json_per_line) in enumerate(data):
#             sub_dataset[f"person number: {json_number}"] = js.loads(json_per_line)
    
#     datasets.append(sub_dataset)


# chiters_activity_tensor = np.zeros(shape=(
#     len(datasets[0]),
#     len(datasets[0]["steps"]["samples"]),
#     len(datasets[0]["steps"]["samples"][0]) + 3
# ))

# print(chiters_activity_tensor)

A = np.random.normal(0.34, 12.34, (30, 10))
added_array = np.asarray([rd.choice([1, 0]) for _ in range(10)])
print(added_array.T.shape)
appended_A = np.c_[A.T, added_array]
shuffle_A = np.random.permutation(appended_A)
print(appended_A)
print(shuffle_A)



