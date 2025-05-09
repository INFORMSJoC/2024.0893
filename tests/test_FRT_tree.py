import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

import initialization
import numpy as np
import uav_initializer
from FRT_tree import FRTtree, Node

a = Node(None, {0, 1}, 1, 0)
print(a.parent)
print(a.children)
print(a.radius)
print(a.level)


class TestFRTTree:
    def test_node(self):
        node = Node(None, {0, 1}, 1, 0)
        assert node.parent is None

    def test_smallest_tree(self):
        t = FRTtree({0, 1}, np.array([[0, 1], [1, 0]]))
        assert t.root.locations_contained == {0, 1}
        assert len(t.node_list) == 3
        assert t.edge_list == [(0, 1), (0, 2)]

    def test_second_smallest_tree(self):
        t = FRTtree({0, 1}, np.array([[0, 2], [2, 0]]))

        assert len(t.node_list) == 5
        assert sorted(t.edge_list) == sorted([(0, 1), (1, 2), (0, 3), (3, 4)])


class TestFRTTreeOnUAV:
    n = 8
    adjacency_matrix = uav_initializer.create_uav_adjaceny_matrix(n, 1, 4, 10)
    dist_matrix, predecessors = initialization.create_distance_matrix(adjacency_matrix)
    locations = set(range(2 * n**2 + 1))
    T = FRTtree(locations, dist_matrix)

    def test_max_distance(self):
        assert (
            self.T.original_cost[127, 128]
            == self.T.original_cost[128, 127]
            == self.T.original_cost[71, 120] + 1
            == self.T.original_cost[120, 71] + 1
            == 35
        ), "Max distance of problem is 34"

        assert self.T.max_distance == 35, "Max distance is correct at 34"

    def test_delta(self):
        assert self.T.log_2_delta == 7, "log_2_delta for example is 7"
        assert self.T.delta == 2**7, "delta for problem is 128"
        assert (
            self.T.delta >= 2 * self.T.max_distance
        ), "delta is larger than 2*max_distance"
        assert (
            2 ** (self.T.log_2_delta - 1) < 2 * self.T.max_distance
        ), "delta is the smallest power of 2 larger than 2*max_distance"
        assert (
            2 ** (self.T.log_2_delta) >= 2 * self.T.max_distance
        ), "delta is larger than 2*max_distance"

    def test_root(self):
        assert (
            self.T.root.locations_contained == self.locations
        ), "Root contains all locations"
        assert (
            self.T.root.radius >= self.T.max_distance
        ), "Root has radius that covers all nodes"

    def test_root_radius_large(self):
        for location in self.locations:
            assert (
                self.T.epsilon_neighbors(location, self.T.root.radius) == self.locations
            ), "Root radius covers all locations"

    def test_epsilon_neighbor(self):
        assert self.T.epsilon_neighbors(0, 0.9) == {0}
        assert self.T.epsilon_neighbors(1, 1) == {0, 1, 2, 9}
        assert self.T.epsilon_neighbors(83, 3) == {83}
        assert self.T.epsilon_neighbors(27, 2) == {
            18,
            19,
            20,
            26,
            27,
            28,
            34,
            35,
            36,
            11,
            25,
            29,
            43,
        }

    def test_leaves_count(self):
        assert len(self.T.leaves) == len(
            self.T.location_set
        ), "Number of leaves is equal to number of locations"

    def test_leaves_vs_level_zero(self):
        assert len(self.T.leaves) == len(
            self.T.level_i_sets[0]
        ), "Number of leaves is equal to number of level zero sets"

    def test_level_contains_all_nodes(self):
        for i in range(self.T.log_2_delta):
            assert set().union(
                *[S.locations_contained for S in self.T.level_i_sets[i]]
            ) == set(range(0, 129)), "Level contains all nodes"

    def test_nonzero_intersection(self):
        for i in range(self.T.log_2_delta):
            assert (
                set().intersection(
                    *[S.locations_contained for S in self.T.level_i_sets[i]]
                )
                == set()
            ), "No intersection of locations"

    def test_decreasing_sets(self):
        sizes = [len(S) for S in self.T.level_i_sets]
        assert sizes == sorted(sizes, reverse=True), "Sets are decreasing in size"

    def test_child_contains_parents(self):
        def _recurs(node):
            if node.level > 0:
                children_items = [child.locations_contained for child in node.children]
                assert node.locations_contained == (
                    set().union(*children_items)
                ), "Children and parent share the same location set"

            for child in node.children:
                assert (
                    node.level - 1 == child.level
                ), "Child level is one less than parent level"
                _recurs(child)

        root_ = self.T.root
        _recurs(root_)

    def test_radius_correctly_assigned(self):
        for level in range(self.T.log_2_delta):
            assert len(self.T.level_i_sets[level]) > 0, "Level is not empty"
            for node in self.T.level_i_sets[level]:
                assert (
                    node.radius == self.T._radius_list[node.level]
                ), "Radius is correctly assigned"
                for loc_ in node.locations_contained:
                    assert node.locations_contained.issubset(
                        self.T.epsilon_neighbors(node.center, node.radius)
                    ), "Radius covers all locations"
                if node.parent is not None:
                    assert (
                        node.radius <= node.parent.radius
                    ), "Radius is smaller than parent radius"

    def test_leaf_correctly_stored(self):
        for i in range(len(self.locations)):
            assert (
                i == self.T.leaves[i].item  # type: ignore
            ), "Leaf is correctly stored"

    def test_tree_metric_larger(self):
        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                tree_dist = self.T.compute_tree_distance(i, j)
                assert (
                    tree_dist >= self.T.original_cost[i, j]
                ), "Tree metric is larger than original cost"
                if self.T.original_cost[i, j] > 0:
                    assert tree_dist / self.T.original_cost[i, j] <= 2 * 2 ** (
                        self.T.log_2_delta
                    ), "Tree metric does not explode"

    def test_edge_count(self):
        assert (
            len(self.T.edge_list) == self.T.number_of_nodes - 1
        ), "Edge count is correct"

    def test_edge_weights(self):
        for level in range(len(self.T.level_i_sets)):
            for node in self.T.level_i_sets[level]:
                assert (
                    node.cost_to_children == 2**level
                ), "Edge weights are not correct"
        assert (
            self.T.level_i_sets[0][0].cost_to_children == 1
        ), "Edge weights are not correct"
