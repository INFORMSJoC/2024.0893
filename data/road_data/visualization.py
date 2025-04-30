"""Script contains code that generates CSV file. 
These files are used in QGIS to visualize the road network.
"""

import numpy as np
import pandas as pd

coordinates = pd.read_pickle("data/road_data/node_coordinates.pkl")
coordinates.to_csv("data/road_data/node_coordinates.csv", index=False)

adjacency = np.load("data/road_data/adjacency_matrix.npy")
edge_list = []
number_of_edges = 0
for i in range(adjacency.shape[0]):
    for j in range(i, adjacency.shape[1]):
        if adjacency[i, j]:
            edge_list.append((*coordinates.iloc[i, 0:2].to_list(), number_of_edges))
            edge_list.append((*coordinates.iloc[j, 0:2].to_list(), number_of_edges))
            number_of_edges += 1
edge_list = pd.DataFrame(edge_list, columns=["x", "y", "edge_id"])
edge_list.to_csv("data/road_data/edge_list.csv", index=False)

sensing_matrix = np.load("data/road_data/sensing_matrix_70_50.npy")
item_list = []
print(sensing_matrix.shape)
for i in range(sensing_matrix.shape[0]):
    for j in range(sensing_matrix.shape[1]):
        if sensing_matrix[i, j]:
            item_list.append((*coordinates.iloc[i, 0:2].to_list(), j))
group_lsit = pd.DataFrame(item_list, columns=["x", "y", "group_id"])
group_lsit.to_csv("data/road_data/group_list.csv", index=False)
