import logging

logger = logging.getLogger(__name__)

import logging

import numpy as np
import numpy.typing as npt
import os

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(dir_path, "data/road_data")


def load_road_sensing_matrix(file_name) -> npt.NDArray[np.bool_]:
    sensing_matrix = np.load(f"{file_path}/{file_name}.npy")
    return sensing_matrix


def load_road_adjacency_matrix() -> npt.NDArray:
    adjacency_matrix = np.load("data/road_data/adjacency_matrix.npy")

    return adjacency_matrix


def load_road_distance_matrix() -> npt.NDArray:
    distance_matrix = np.load("data/road_data/graph_distance.npy")

    return distance_matrix


if __name__ == "__main__":
    print(load_road_adjacency_matrix())
    print(load_road_distance_matrix())
