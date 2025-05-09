"""
Module approximately solves the Group Steiner Tree problem using the deterministic GKR rounding algorithm.
"""

from __future__ import annotations

import logging
import math
import typing
from collections import deque

import group_steiner_LP_relaxation
import initialization
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from FRT_tree import FRTtree

if typing.TYPE_CHECKING:
    from FRT_tree import Node
    from grouping import Group

module_logger = logging.getLogger(__name__)


class GroupSteinerTree:
    def __init__(
        self,
        location_set: set,
        original_cost: npt.NDArray[np.float_],
        groups: list[Group],
    ) -> None:
        self.groups = groups
        self.grid_root_index = max(location_set)

        self._hierarchical_tree = FRTtree(location_set, original_cost)
        self.grid_root_node = self._hierarchical_tree.leaves[self.grid_root_index]

        self.leaves: list[Node] = self._hierarchical_tree.leaves  # type: ignore
        self.node_list = self._hierarchical_tree.node_list
        self._preprocess()

    @property
    def number_of_groups(self) -> int:
        return len(self.groups)

    def _preprocess(self):
        module_logger.info("Starting preprocess")
        self._create_tree_adjacency_matrix()
        if __debug__:
            np.savetxt("hierarchical_tree.txt", self.adjacency_matrix, fmt="%d")
        self._smoothen_tree()
        if __debug__:
            np.savetxt(
                "hierarchical_tree_smoothened.txt", self.adjacency_matrix, fmt="%d"
            )
            assert len(self.node_list) == len(
                self.adjacency_matrix
            ), "Correctly removed nodes"
        self._node_renumbering()
        self._reroot_to_last_leaf()
        self._update_node_parent_and_children()
        self._assign_depth()
        self._create_lowest_common_ancestor_dictionary()

    def _create_tree_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros(
            (
                self._hierarchical_tree.number_of_nodes,
                self._hierarchical_tree.number_of_nodes,
            ),
        )
        for level_i_set in self._hierarchical_tree.level_i_sets:
            for node in level_i_set:
                for child in node.children:
                    self.adjacency_matrix[node.id, child.id] = node.cost_to_children
                    self.adjacency_matrix[child.id, node.id] = node.cost_to_children
        assert (
            self.adjacency_matrix == self.adjacency_matrix.T
        ).all, "Adjacency matrix is not symmetric"

    def _smoothen_tree(self):
        """
        This method smoothen tree by removing degree 2 edges.
        """
        # degrees = np.count_nonzero(self.adjacency_matrix, axis=1)
        # allow_list = [True] * self.number_of_nodes

        # edge_weight_accumulation = 0
        # for i, degree in enumerate(degrees):
        #     if degree == 2 and i > 0:
        #         parent_idx, child_idx = np.nonzero(self.adjacency_matrix[i, :])[0]
        #         allow_list[i] = False

        mat = self.adjacency_matrix.copy()
        allow_list = np.full(len(mat), True)

        for i in range(1, len(mat)):
            row = mat[i, :]
            degree = np.count_nonzero(row)
            if degree != 2:
                continue
            source, target = np.nonzero(row)[0]
            assert (
                source < i and i < target
            ), "DFS numbering should ensure that the parent is always smaller than the child"
            mat[source, target] = np.sum(row)
            mat[target, source] = np.sum(row)
            mat[source, i] = 0
            mat[i, target] = 0
            mat[target, i] = 0
            mat[i, source] = 0
            allow_list[i] = False
        mat = mat[np.ix_(allow_list, allow_list)]
        # mat = np.triu(mat) + np.triu(mat).T

        assert np.all(mat == mat.T), "Adjacency matrix should be symmetric"
        assert (
            mat.diagonal() == 0
        ).all(), "Adjacency matrix should not have self loops"
        assert (
            mat.sum() == self.adjacency_matrix.sum()
        ), f"{mat.sum()} != {self.adjacency_matrix.sum()}, diff = {self.adjacency_matrix.sum() - mat.sum()}"

        assert len(allow_list) == len(
            self.node_list
        ), "Graph should still have the same total edge weights."

        self.node_list = [
            (self.node_list[i]) for i in range(len(self.node_list)) if allow_list[i]
        ]

        if __debug__:
            degrees_of_old_tree = np.count_nonzero(self.adjacency_matrix, axis=1)[1:]
            degrees_of_new_tree = np.count_nonzero(mat, axis=1)[1:]
            filtered_degrees_of_old_tree = np.array(
                [degree for degree in degrees_of_old_tree if degree != 2]
            )
            assert (
                filtered_degrees_of_old_tree == degrees_of_new_tree
            ).all(), "Smoothing only removes degree 2 nodes"

            old_shortest_path = sp.csgraph.shortest_path(
                sp.csr_matrix(self.adjacency_matrix)
            )
            new_shortest_path = sp.csgraph.shortest_path(sp.csr_matrix(mat))

            # We don't expect the path length between nodes to change.
            for i, first_node in enumerate(self.node_list):
                for j, second_node in enumerate(self.node_list):
                    assert (
                        old_shortest_path[first_node.id, second_node.id]
                        == new_shortest_path[i, j]
                    ), f"Shortest path between {first_node.id} and {second_node.id} is different"
        self.adjacency_matrix = mat

    def _node_renumbering(self):
        for i, node in enumerate(self.node_list):
            node.id = i

    def _reroot_to_last_leaf(self):
        """
        This function creates a list of edges from an adjacency matrix.
        """
        self.rerooted_edge_list = []
        self.rerooted_parent_dict = {}
        self.rerooted_children_dict = {
            node_id: [] for node_id in range(len(self.adjacency_matrix))
        }
        frontier = [self.grid_root_node.id]
        self.rerooted_parent_dict[self.grid_root_node.id] = None

        while frontier:
            node_idx = frontier.pop()
            for i in range(len(self.adjacency_matrix)):
                if (
                    self.adjacency_matrix[node_idx, i] > 0
                    and i not in self.rerooted_parent_dict
                ):
                    self.rerooted_edge_list.append((node_idx, i))
                    self.rerooted_parent_dict[i] = node_idx
                    self.rerooted_children_dict[node_idx].append(i)
                    frontier.append(i)
        assert (
            len(self.rerooted_edge_list) == len(self.adjacency_matrix) - 1
        ), "Number of edges should be equal to number of nodes - 1"
        assert np.count_nonzero(self.adjacency_matrix) == 2 * len(
            self.rerooted_edge_list
        ), "Number of edges (directed) half the number of non-zero entries in the adjacency matrix."

        assert sum(
            [
                len(children_list)
                for children_list in self.rerooted_children_dict.values()
            ]
        ) == len(
            self.rerooted_edge_list
        ), "Total number of children across all nodes is the number of edges"

    def _update_node_parent_and_children(self):
        self.root = self.grid_root_node

        # We want to ultimately remove it, but we store it purely for checking.
        root_check = self.leaves.pop()
        assert root_check == self.root
        for node in self.node_list:
            if node.id != self.grid_root_node.id:
                node.parent = self.node_list[self.rerooted_parent_dict[node.id]]
            else:
                node.parent = None
            node.children = [
                self.node_list[node_id]
                for node_id in self.rerooted_children_dict[node.id]
            ]

    def _assign_depth(self):
        def recurs(node):
            for child in node.children:
                child.depth = node.depth + 1
                child.parent = node
                recurs(child)

        self.grid_root_node.depth = 0

        recurs(self.grid_root_node)
        self.height = max(leaf.depth for leaf in self.leaves)

    def get_tour(self):
        (
            vals_x,
            vals_y,
            vals_f,
            lp_model,
        ) = group_steiner_LP_relaxation.solve_linear_program(
            self.adjacency_matrix,
            self.groups,
            self.leaves,
            self.grid_root_node,
            self.rerooted_edge_list,
            self.rerooted_parent_dict,
            self.rerooted_children_dict,
        )
        # module_logger.debug(
        #     f"LP solved: \n\t X: {self.find_non_zero_dict_entries(vals_x)} \n"
        #     f"\t Y: {self.find_non_zero_dict_entries(vals_y)} \n"
        #     f"\t F: {self.find_non_zero_dict_entries(vals_f)}"
        # )
        self.x_lp = vals_x
        self.y_lp = vals_y
        self.f_lp = vals_f

        assert all(
            [
                math.isclose(
                    vals_x[leaf.parent.id, leaf.id],
                    max(
                        [
                            vals_f[(leaf.parent.id, leaf.id), group_idx]
                            for group_idx in range(self.number_of_groups)
                        ]
                    ),
                )
                for leaf in self.leaves
                if leaf.parent is not None
            ]
        ), "x value at the leaf is equal to at least one f value"
        assert lp_model.objVal == self._compute_current_cost(vals_x)
        self.lp_model = lp_model

        self._round()

    def _create_lowest_common_ancestor_dictionary(self):
        """Find the lowest common ancestor between all pairs of leaves"""
        self.lowest_common_ancestor_dict = {}
        for i in range(len(self.leaves)):
            u_node = self.leaves[i]
            path_to_root = set()
            curr_node = u_node
            # Create a leaf to root path
            while curr_node is not None:
                path_to_root.add(curr_node.id)
                curr_node = curr_node.parent
            for j in range(i + 1, len(self.leaves)):
                v_node = self.leaves[j]
                curr_node = v_node
                # Find the lowest element that is within the leaf to root path of v.
                while curr_node is not None:
                    if curr_node.id in path_to_root:
                        assert (
                            u_node is not None and v_node is not None
                        ), "nodes not correctly stored."
                        self.lowest_common_ancestor_dict[
                            (u_node.item, v_node.item)
                        ] = curr_node
                        self.lowest_common_ancestor_dict[
                            (v_node.item, u_node.item)
                        ] = curr_node
                        break
                    curr_node = curr_node.parent
                else:
                    raise (ValueError("No common ancestor found"))

    def retrieve_parent_edge(self, node: Node) -> tuple[int, int] | None:
        """For node u, retrieve the edge (u.parent, u))"""
        if node.parent is not None:
            return (node.parent.id, node.id)

    def _create_tree_when_set_0(
        self,
        node: Node,
        X_bar: dict[tuple[int, int], float],
        f_bar: dict[tuple[tuple[int, int], int], float],
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[tuple[int, int], int], float]]:
        def recurs(node, X_bar, f_bar):
            curr_edge = self.retrieve_parent_edge(node)
            X_bar[curr_edge] = 0
            if node.children:
                for child in node.children:
                    recurs(child, X_bar, f_bar)
            else:
                for i in range(len(self.groups)):
                    if (curr_edge, i) in f_bar:
                        f_bar[curr_edge, i] = 0

        X_bar_0 = X_bar.copy()
        f_bar_0 = f_bar.copy()
        recurs(node, X_bar_0, f_bar_0)
        assert all(
            [
                math.isclose(
                    X_bar_0[leaf.parent.id, leaf.id],
                    max(
                        [
                            f_bar_0[(leaf.parent.id, leaf.id), group_idx]
                            for group_idx in range(self.number_of_groups)
                        ]
                    ),
                )
                for leaf in self.leaves
                if leaf.parent is not None
            ]
        ), "x value at the leaf is equal to at least one f value"
        return X_bar_0, f_bar_0

    def _create_tree_when_set_1(
        self,
        node: Node,
        X_bar: dict[tuple[int, int], float],
        f_bar: dict[tuple[tuple[int, int], int], float],
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[tuple[int, int], int], float]]:
        def recurs(node: Node, X_bar: dict, f_bar: dict, x_e: float):
            curr_edge = self.retrieve_parent_edge(node)
            curr = X_bar[curr_edge]
            conditional_probability = curr / x_e

            # Handle floating point issues
            if math.isclose(conditional_probability, 1):
                conditional_probability = 1
            X_bar[self.retrieve_parent_edge(node)] = conditional_probability
            if node.children:
                for child in node.children:
                    recurs(child, X_bar, f_bar, x_e)
            else:
                for i in range(len(self.groups)):
                    if (curr_edge, i) in f_bar:
                        # Handle floating point issues
                        conditional_probability = f_bar[curr_edge, i] / x_e
                        if math.isclose(conditional_probability, 1):
                            conditional_probability = 1
                        f_bar[curr_edge, i] = conditional_probability

        X_bar_1 = X_bar.copy()
        f_bar_1 = f_bar.copy()
        parent_edge_of_e = self.retrieve_parent_edge(node)
        assert parent_edge_of_e is not None, "parent edge of e is None"
        x_e = X_bar_1[parent_edge_of_e]
        recurs(node, X_bar_1, f_bar_1, x_e)

        assert all(
            [
                math.isclose(
                    X_bar_1[leaf.parent.id, leaf.id],
                    max(
                        [
                            f_bar_1[(leaf.parent.id, leaf.id), group_idx]
                            for group_idx in range(self.number_of_groups)
                        ]
                    ),
                )
                for leaf in self.leaves
                if leaf.parent is not None
            ]
        ), "x value at the leaf is equal to at least one f value"
        return X_bar_1, f_bar_1

    def _compute_current_cost(self, X_bar: dict[tuple[int, int], float]) -> float:
        return sum([X_bar[edge] * self.adjacency_matrix[edge] for edge in X_bar])

    def _compute_current_profit(
        self,
        X_bar: dict[tuple[int, int], float],
        f_bar: dict[tuple[tuple[int, int], int], float],
    ) -> float:
        profit = 0

        for group_idx in range(len(self.groups)):
            for v_location_idx in self.groups[group_idx]:
                pi_v = self.retrieve_parent_edge(self.leaves[v_location_idx])

                assert pi_v is not None, "pi_v is None"
                x_pi_v = f_bar[(pi_v), group_idx]
                profit += self.groups[group_idx].group_weight * x_pi_v
                for u_location_idx in self.groups[group_idx]:
                    pi_u = self.retrieve_parent_edge(self.leaves[u_location_idx])
                    assert pi_u is not None, "pi_u is None"
                    x_pi_u = f_bar[(pi_u), group_idx]
                    lowest_common_ancestor_node = self.lowest_common_ancestor_dict.get(
                        (u_location_idx, v_location_idx)
                    )
                    if lowest_common_ancestor_node is not None:
                        lowest_common_ancestor_edge = self.retrieve_parent_edge(
                            lowest_common_ancestor_node
                        )
                    else:
                        lowest_common_ancestor_edge = None

                    if lowest_common_ancestor_edge is not None:
                        x_ances = X_bar[lowest_common_ancestor_edge]
                    elif u_location_idx == v_location_idx:
                        x_ances = x_pi_u
                    else:
                        x_ances = 1

                    H = self.height
                    if x_ances > 0:
                        profit -= (
                            self.groups[group_idx].group_weight
                            * (1 / (2 * H))
                            * x_pi_v
                            * (x_pi_u / x_ances)
                        )
        return profit

    def _compute_best_edge_assignment(
        self,
        node: Node,
        X_bar: dict[tuple[int, int], float],
        f_bar: dict[tuple[tuple[int, int], int], float],
    ) -> tuple[
        dict[tuple[int, int], float],
        dict[tuple[tuple[int, int], int], float],
        float,
    ]:
        X_bar_0, f_bar_0 = self._create_tree_when_set_0(node, X_bar, f_bar)
        X_bar_1, f_bar_1 = self._create_tree_when_set_1(node, X_bar, f_bar)

        cost_0 = self._compute_current_cost(X_bar_0)
        cost_1 = self._compute_current_cost(X_bar_1)

        profit_0 = self._compute_current_profit(X_bar_0, f_bar_0)
        profit_1 = self._compute_current_profit(X_bar_1, f_bar_1)

        if __debug__:
            curr_edge = self.retrieve_parent_edge(node)

            prev_cost = self._compute_current_cost(X_bar)
            prev_profit = self._compute_current_profit(X_bar, f_bar)
            assert curr_edge is not None, "curr_edge is None"
            assert (
                cost_0 <= prev_cost <= cost_1
            ), f"{cost_0=} <= {prev_cost=} <= {cost_1=} is not True"
            assert math.isclose(
                prev_cost,
                X_bar[curr_edge] * cost_1 + (1 - X_bar[curr_edge]) * cost_0,
            ), f"{prev_cost=} != {X_bar[curr_edge]=} * {cost_1=} + {(1 - X_bar[curr_edge])=} * {cost_0=}"
            assert (
                profit_0 <= prev_profit <= profit_1
            ), f"{profit_0=} <= {prev_profit=} <= {profit_1=} is not True"
            assert math.isclose(
                prev_profit,
                X_bar[curr_edge] * profit_1 + (1 - X_bar[curr_edge]) * profit_0,
            ), (
                f"{prev_profit=} != {X_bar[curr_edge]=} * {profit_1=} + {(1 - X_bar[curr_edge])=} * {profit_0=}"
                f"= {X_bar[curr_edge] * profit_1 + (1 - X_bar[curr_edge]) * profit_0}"
            )

        ratio_0 = cost_0 / profit_0 if profit_0 > 0 else np.inf
        ratio_1 = cost_1 / profit_1 if profit_1 > 0 else np.inf
        curr_edge = self.retrieve_parent_edge(node)
        module_logger.debug(
            f"Values when rounded edge({curr_edge}) is rounded to:\n"
            f"\t {cost_0=}, {profit_0=}, {ratio_0=}\n"
            f"\t {cost_1=}, {profit_1=}, {ratio_1=}"
        )
        if ratio_0 <= ratio_1:
            module_logger.debug(
                f"Edge {curr_edge} rounded to 0 resulting in:\n"
                f"\t cost={cost_0}, profit={profit_0}, ratio={ratio_0}"
            )
            return X_bar_0, f_bar_0, ratio_0

        else:
            module_logger.debug(
                f"Edge {curr_edge} rounded to 1 resulting in:\n"
                f"\t cost={cost_1}, profit={profit_1}, ratio={ratio_1}"
            )
            return (
                X_bar_1,
                f_bar_1,
                ratio_1,
            )

    def _round(self):
        X_bar = self.x_lp.copy()
        f_bar = {}

        # initialize f_bar only for relevant entries (those connected to the leaves)
        for group_idx in range(len(self.groups)):
            for leaf in self.leaves:
                edge = self.retrieve_parent_edge(leaf)
                f_bar[edge, group_idx] = self.f_lp[edge, group_idx]

        frontier = deque()
        frontier.append(self.grid_root_node)
        H = self.height

        lowest_ratio_found = np.inf
        curr_ratio = lowest_ratio_found

        while frontier:
            curr = frontier.pop()
            for child_node in curr.children:
                curr_edge = self.retrieve_parent_edge(child_node)
                assert curr_edge is not None, "curr_edge is None"
                if 0 < X_bar[curr_edge]:
                    module_logger.debug(
                        f"Rounding edge: {curr_edge} with LP value: {X_bar[curr_edge]}"
                    )
                    module_logger.debug(
                        f"X_bar before rounding: {self.find_non_zero_dict_entries(X_bar)}"
                    )
                    if X_bar[curr_edge] < 1:
                        rounded_edge_val = self._compute_best_edge_assignment(
                            child_node, X_bar, f_bar
                        )

                        (
                            X_bar,
                            f_bar,
                            curr_ratio,
                        ) = rounded_edge_val
                        module_logger.debug(
                            f"X_bar after rounding: {self.find_non_zero_dict_entries(X_bar)}"
                        )
                    assert (
                        curr_ratio <= lowest_ratio_found
                    ), f"{curr_ratio=} > {lowest_ratio_found=}"
                    lowest_ratio_found = curr_ratio
                    frontier.append(child_node)
        self.ratio_from_rounding = curr_ratio
        self.X_bar = self.find_non_zero_dict_entries(X_bar)
        self.f_bar = self.find_non_zero_dict_entries(f_bar)
        self.selected_edges = [key for key in X_bar if X_bar[key] > 0.5]
        selected_head = [v for (_, v) in self.selected_edges]
        self.selected_leaves = [
            leaf for leaf in self.leaves if leaf.id in selected_head
        ]
        module_logger.debug(self.find_non_zero_dict_entries(self.X_bar))
        module_logger.debug(self.selected_edges)
        module_logger.debug(self.selected_leaves)

    def create_eulerian_tour(self):
        tour = []
        index_of_leaf_nodes = {leaf.id for leaf in self.leaves}
        tour_on_grid = [self.grid_root_index]
        root_idx = self.grid_root_node.id
        selected_edges_dict = {}
        for (u, v), value in self.X_bar.items():
            if u in selected_edges_dict:
                selected_edges_dict[u].append((v, value))
            else:
                selected_edges_dict[u] = [(v, value)]

        def dfs(node_idx: int):
            tour.append(node_idx)
            if node_idx in index_of_leaf_nodes:
                grid_index = self.node_list[node_idx].item
                assert grid_index is not None, "node does not seem to be a leaf."
                tour_on_grid.append(grid_index)
            for neighbor in selected_edges_dict.get(node_idx, []):
                selected_edges_dict[node_idx].remove(neighbor)
                dfs(neighbor[0])
                tour.append(node_idx)

        dfs(root_idx)
        tour_on_grid.append(self.grid_root_index)
        return tour, tour_on_grid

    def find_non_zero_dict_entries(self, dict):
        return {key: value for (key, value) in dict.items() if value > 0}


if __name__ == "__main__":
    import grouping

    l = 8
    c_top = 1
    c_bot = 4
    height = 10

    n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
        "uav", l=l, c_top=c_top, c_bot=c_bot, height=height
    )

    (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    ) = initialization.initialize_variables("uav", l=l)

    group_list = grouping.create_groups(
        locations, compatible_scenarios, probability_distribution, sensing_matrix
    )
    print(group_list)
    tree = GroupSteinerTree(locations, distance_matrix, group_list)
    print("====================Solving LP====================")
    tree.get_tour()
    print(tree)
    print(tree.create_eulerian_tour())
    print(tree.create_eulerian_tour())
