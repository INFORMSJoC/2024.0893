import os
import sys

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../IPP/road_data_creation"
)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

import numpy as np
from FRT_tree import FRTtree, Node
import sensing_matrix_creator


class TestCascade:
    adjacency_distance = np.load("data/road_data/adjacency_matrix.npy")
    adjacency_matrix = adjacency_distance > 0

    def test_cascade_one(self):
        reachable = sensing_matrix_creator.independent_cascade(
            self.adjacency_matrix, 1, 0
        )
        assert len(reachable) == len(self.adjacency_matrix)

    def test_cascade_zero(self):
        for i in range(len(self.adjacency_matrix)):
            reachable = sensing_matrix_creator.independent_cascade(
                self.adjacency_matrix, 0, i
            )
            assert len(reachable) == 1
