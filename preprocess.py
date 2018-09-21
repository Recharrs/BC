import os
import json
import pickle

import matplotlib.pyplot as plt

# state_action_pair = []

# root = "Dataset"
# for filename in os.listdir(root):
#     file_path = os.path.join(root, filename)
#     with open(file_path, "r") as file:
#         data = json.load(file)
#         state = data["state"]
#         action = data["action"]
#         for step in range(len(action)): 
#             state_action_pair.append([state[step], action[step]])

# print(len(state_action_pair))
# with open("traj/state_action_pair.pickle", "wb") as file:
#     pickle.dump(state_action_pair, file)

with open("logger.txt", "r") as file:
    lines = file.read().splitlines()

    data_size = []
    dists = []

    for line in lines:
        data = line.split(" ")
        data_size.append(int(data[1][:-1]))
        dists.append(float(data[3]))
    
    plt.xlabel("# of data")
    plt.ylabel("mean distance")

    plt.plot(data_size, dists)
    plt.show()

    with open("logger_cooked.pickle", "wb") as file_out:
        data_out = {
            "data_size": data_size,
            "dists": dists,
        }
        pickle.dump(data_out, file_out)
