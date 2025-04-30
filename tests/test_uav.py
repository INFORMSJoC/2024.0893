import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

import initialization
import uav_initializer


class TestUAVdistance:
    def test_2d_grid_adjacency_size_8(self):
        grid8 = uav_initializer.create_2d_grid_adjacency(8, 8)
        assert grid8.shape == (64, 64)
        for i in range(8):
            for j in range(8):
                assert grid8[i * 8 + j, i * 8 + j] == 0
                if i > 0:
                    assert grid8[i * 8 + j, (i - 1) * 8 + j] == 1
                if i < 7:
                    assert grid8[i * 8 + j, (i + 1) * 8 + j] == 1
                if j > 0:
                    assert grid8[i * 8 + j, i * 8 + j - 1] == 1
                if j < 7:
                    assert grid8[i * 8 + j, i * 8 + j + 1] == 1


class TestInitialization:
    def test_problem_matrix_size_8(self):
        assert uav_initializer.create_uav_adjaceny_matrix(8, 4, 1, 10).shape == (
            129,
            129,
        )

    def test_problem_matrix_size_2(self):
        assert uav_initializer.create_uav_adjaceny_matrix(2, 4, 1, 10).shape == (9, 9)

    def test_symmetry(self):
        board = uav_initializer.create_uav_adjaceny_matrix(8, 4, 1, 10)
        assert np.array_equal(board, board.T)

    def test_problem_mini(self):
        board = np.array(
            [
                [0, 4, 4, 0, 10, 0, 0, 0, 0],
                [4, 0, 0, 4, 0, 10, 0, 0, 0],
                [4, 0, 0, 4, 0, 0, 10, 0, 0],
                [0, 4, 4, 0, 0, 0, 0, 10, 0],
                [10, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 10, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 10, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 10, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        assert np.array_equal(
            uav_initializer.create_uav_adjaceny_matrix(2, 4, 1, 10), board
        )


class TestNeighboring:
    def test_neighbors(self):
        board = uav_initializer.create_uav_adjaceny_matrix(8, 1, 4, 10)
        assert uav_initializer.generate_neighborhood(0, board) == {1, 8, 64}
        assert uav_initializer.generate_neighborhood(1, board) == {0, 2, 9, 65}
        assert uav_initializer.generate_neighborhood(10, board) == {
            2,
            9,
            11,
            18,
            74,
        }


class TestDistance:
    def test_distance(self):
        board = uav_initializer.create_uav_adjaceny_matrix(8, 1, 4, 10)
        dist_matrix, predecessors = initialization.create_distance_matrix(board)
        assert dist_matrix[0, 1] == 1
        assert dist_matrix[0, 8] == 1
        assert dist_matrix[0, 63] == 14
        assert dist_matrix[0, 64] == 10
        assert dist_matrix[0, 65] == 10 + 1
        assert dist_matrix[0, 127] == 14 + 10

    def test_root_distance(self):
        board = uav_initializer.create_uav_adjaceny_matrix(8, 1, 4, 10)
        dist_matrix, predecessors = initialization.create_distance_matrix(board)
        assert dist_matrix[64, 128] == dist_matrix[128, 64] == 1
        assert dist_matrix[127, 128] == dist_matrix[64, 127] + 1
        assert dist_matrix[128, 0] == dist_matrix[0, 128] == 11

    def test_symmtry(self):
        board = uav_initializer.create_uav_adjaceny_matrix(8, 1, 4, 10)
        dist_matrix, predecessors = initialization.create_distance_matrix(board)
        assert np.array_equal(dist_matrix, dist_matrix.T)
