import os
import sys
from unittest import mock

import pytest
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

from typing import Any
import grouping
import numpy as np
from group_steiner_LP_relaxation import solve_linear_program


def build_tree_from_list(tree_edge):
    max_node_value = max([max(u, v) for u, v, w in tree_edge])
    adjacency_matrix = np.zeros((max_node_value + 1, max_node_value + 1))
    parent_dict = {i: None for i in range(max_node_value + 1)}
    child_dict = {i: [] for i in range(max_node_value + 1)}
    edge_list = []
    for u, v, w in tree_edge:
        adjacency_matrix[u, v] = w
        adjacency_matrix[v, u] = w
        parent_dict[v] = u
        child_dict[u].append(v)
        edge_list.append((u, v))

    return adjacency_matrix, parent_dict, child_dict, edge_list


@pytest.fixture
def leaves():
    leaves_idx_list = [2]
    leaves = []
    for i, idx in enumerate(leaves_idx_list):
        new_leaf = mock.Mock()
        new_leaf.id = idx
        new_leaf.item = i
        leaves.append(new_leaf)
    yield leaves


@pytest.fixture
def split_leaves():
    leaves_idx_list = [2, 3]
    leaves = []
    for i, idx in enumerate(leaves_idx_list):
        new_leaf = mock.Mock()
        new_leaf.id = idx
        new_leaf.item = i
        leaves.append(new_leaf)
    return leaves


@pytest.fixture
def root_node():
    root_node = mock.Mock()
    root_node.id = 0

    yield root_node


class TestGroupSteinerLPRelaxation:
    def test_trivial(self, leaves: list[Any], root_node: Mock):
        tree_edge = [(0, 1, 2), (1, 2, 3)]
        adjacency_matrix, parent_dict, child_dict, edge_list = build_tree_from_list(
            tree_edge
        )
        groups = [grouping.Group({0}, 1)]
        vals_X, vals_y, vals_flows, m = solve_linear_program(
            adjacency_matrix,
            groups,
            leaves,
            root_node,
            edge_list,
            parent_dict,
            child_dict,
        )
        assert vals_X == {(0, 1): 1.0, (1, 2): 1.0}
        assert vals_y == {0: 1.0}
        assert vals_flows == {((0, 1), 0): 1.0, ((1, 2), 0): 1.0}
        groups = [grouping.Group({0}, 2)]
        vals_X, vals_y, vals_flows, m = solve_linear_program(
            adjacency_matrix,
            groups,
            leaves,
            root_node,
            edge_list,
            parent_dict,
            child_dict,
        )
        assert vals_X == {(0, 1): 1.0, (1, 2): 1.0}
        assert vals_y == {0: 1.0}
        assert vals_flows == {((0, 1), 0): 1.0, ((1, 2), 0): 1.0}
        assert m.objVal == 5

    def test_fork(self, root_node: Mock, split_leaves: list[Any]):
        tree_edge = [(0, 1, 1), (1, 2, 1), (1, 3, 1)]
        adjacency_matrix, parent_dict, child_dict, edge_list = build_tree_from_list(
            tree_edge
        )
        groups = [grouping.Group({0}, 1), grouping.Group({1}, 1)]
        vals_X, vals_y, vals_flows, m = solve_linear_program(
            adjacency_matrix,
            groups,
            split_leaves,
            root_node,
            edge_list,
            parent_dict,
            child_dict,
        )
        assert vals_X == {(0, 1): 0.5, (1, 2): 0.5, (1, 3): 0.5}
        assert vals_y == {0: 0.5, 1: 0.5}
        assert vals_flows == {
            ((0, 1), 0): 0.5,
            ((0, 1), 1): 0.5,
            ((1, 2), 0): 0.5,
            ((1, 2), 1): 0,
            ((1, 3), 0): 0,
            ((1, 3), 1): 0.5,
        }
        assert m.objVal == 1.5


if __name__ == "__main__":
    test = TestGroupSteinerLPRelaxation()
    test = TestGroupSteinerLPRelaxation()
