import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

import star_initializer
import initialization
import sensing


class TestStarDistance:
    def test_graph_size(self):
        smallest = star_initializer.create_star_adjacency_matrix(1, 2)
        assert smallest.shape == (5, 5)
        assert np.count_nonzero(smallest) // 2 == 5 - 1
        eight = star_initializer.create_star_adjacency_matrix(8, 2)
        expected = 8 + 2**8 + 2
        assert eight.shape == (expected, expected)
        assert np.count_nonzero(eight) // 2 == expected - 1

    def test_initialize_data_shape(self):
        n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
            "star", bits=3, d=2
        )
        assert adjacency.shape == (n, n)
        assert distance_matrix.shape == (n, n)
        assert sensing_matrix.shape == (n - 2, 2**3)

    def test_star_adjacency_matrix(self):
        for n in range(5, 8):
            for d in (10, 57):
                adjacency = star_initializer.create_star_adjacency_matrix(n, d)
                distance = initialization.create_distance_matrix(adjacency)[0]
                for i in range(2**n):
                    assert distance[-1, i] == 1
                    for j in range(2**n):
                        if i != j:
                            assert distance[i, j] == 2
                    for j in range(2**n, 2**n + n):
                        assert distance[i, j] == 2 + d

                for i in range(2**n, 2**n + n):
                    assert distance[-2, i] == 1
                    for j in range(2**n, 2**n + n):
                        if i != j:
                            assert distance[i, j] == 2, f"{i}, {j}"
                assert distance[2**n + n, 2**n + n + 1] == d
                assert (distance == distance.T).all()

    def test_star_sensing_matrix(self):
        sensing_matrix = star_initializer.create_star_sensing_matrix(5)
        for i in range(2**5):
            assert sensing.find_eliminated_scenarios(
                [sensing.Observation(i, False)], set(range(2**5)), sensing_matrix
            ) == {i}
            assert sensing.find_eliminated_scenarios(
                [sensing.Observation(i, True)], set(range(2**5)), sensing_matrix
            ) == set(range(2**5)) - {i}
        for i in range(2**5, 2**5 + 5):
            assert (
                len(
                    sensing.find_eliminated_scenarios(
                        [sensing.Observation(i, False)],
                        set(range(2**5)),
                        sensing_matrix,
                    )
                )
                == 2**4
            )
            assert (
                len(
                    sensing.find_eliminated_scenarios(
                        [sensing.Observation(i, True)],
                        set(range(2**5)),
                        sensing_matrix,
                    )
                )
                == 2**4
            )
