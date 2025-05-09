from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import numpy.typing as npt

module_logger = logging.getLogger(__name__)


class Node:
    number_of_nodes = 0

    def __init__(
        self,
        parent: Node | None,
        locations_contained: set[int],
        radius: float,
        level: int,
        center: int | None = None,
        reset_counter: bool = False,
    ) -> None:
        self.parent = parent
        self.children: list[Node] = []
        self.locations_contained = locations_contained
        self.center = center
        self.radius = radius
        self.level = level
        self.depth = level
        if reset_counter:
            Node.number_of_nodes = 0
        self.id = Node.number_of_nodes
        Node.number_of_nodes += 1

    def add_child(self, child: Node):
        self.children.append(child)

    @property
    def item(self) -> int:
        if len(self.locations_contained) != 1:
            raise ValueError("The node is not a singleton node.")
        (item_,) = self.locations_contained
        return item_

    @property
    def cost_to_children(self) -> float:
        return 2**self.level

    def __repr__(self) -> str:
        return f"Node({self.id}, {self.locations_contained}, {self.depth})"

    def __contains__(self, location: int) -> bool:
        return location in self.locations_contained


class FRTtree:
    def __init__(
        self, location_set: set, original_cost: npt.NDArray[np.float_]
    ) -> None:
        self.location_set = location_set
        if np.diagonal(original_cost).any():
            raise ValueError("The diagonal of the cost matrix should be zero.")
        self.original_cost = original_cost

        # The following 3 are just different mapping to retrieve nodes.
        # Leaves allows access to the location on the original grid
        self.leaves: list = [None for x in range(len(self.location_set))]
        # Level i sets allows quick access to nodes at each level
        self.level_i_sets = [[] for _ in range(self.log_2_delta + 1)]
        # Node list is placed such that node_list[i] is the node with id i
        self.node_list = []
        self._random_permutation_pi = np.random.permutation(list(self.location_set))
        self._r_0 = np.random.uniform(1 / 2, 1)
        self._radius_list = np.array(
            [self._r_0 * 2**i for i in range(self.log_2_delta + 1)]
        )
        self._hierarchical_tree_decomposition()

    @property
    def max_distance(self) -> float:
        """
        Gives us the largest distance between any 2 pairs of node.
        """
        return self.original_cost.max()

    @property
    def log_2_delta(self) -> int:
        """
        Smallest exponent of 2 such that 2**i is  larger than 2max d_uv
        """
        return int(np.ceil(np.log2(2 * self.max_distance)))

    @property
    def delta(self) -> float:
        """
        Smallest power of 2 greater than 2max d_uv
        """
        delta = 2**self.log_2_delta
        assert delta / 2 < 2 * self.max_distance <= delta, (
            f"{delta/2} < {2*self.max_distance}" f"<= {delta} not satisfied"
        )
        return delta

    @property
    def number_of_locations(self) -> int:
        return len(self.location_set)

    @property
    def edge_list(self) -> list[tuple[int, int]]:
        edge_list = []

        def _recurs(node: Node):
            for child in node.children:
                edge_list.append((node.id, child.id))
                _recurs(child)

        _recurs(self.root)
        return edge_list

    def _add_to_node_list(self, node):
        self.node_list.append(node)
        assert (
            node.id == len(self.node_list) - 1
        ), f"{node.id=} != {len(self.node_list)}-1"

    def epsilon_neighbors(self, location: int, epsilon: float) -> set[int]:
        """Return the locations that are at most
        epsilon away from the given location"""
        return set(np.where(self.original_cost[location] <= epsilon)[0])

    def _add_to_level_i_sets(self, node: Node) -> None:
        self.level_i_sets[node.level].append(node)

    def _hierarchical_tree_decomposition(self) -> None:
        """Perform the hierarchical tree decomposition"""

        root_ = Node(
            None,
            self.location_set,
            self._radius_list[-1],
            self.log_2_delta,
            reset_counter=True,
        )
        self._add_to_level_i_sets(root_)
        self._add_to_node_list(root_)
        self._tree_recurs(root_)

        assert all(
            node is not None for node in self.leaves
        ), 'All leaves should be of type "Node"'

        self.root = root_
        self.number_of_nodes = Node.number_of_nodes

    def _tree_recurs(self, node: Node) -> None:
        if node.level == 0:
            assert (
                len(node.locations_contained) == 1
            ), "The leaf node should contain only one location"
            assert node.item is not None, "The leaf node is not singleton node"
            assert self.leaves[node.item] is None, "We visited the same leaf twice."
            self.leaves[node.item] = node
        else:
            S = node.locations_contained.copy()
            for j in range(self.number_of_locations):
                if self._random_permutation_pi[j] in S:
                    intersection_ = (
                        self.epsilon_neighbors(
                            self._random_permutation_pi[j],
                            self._radius_list[node.level - 1],
                        )
                        & S
                    )
                    if intersection_:
                        child_node = Node(
                            node,
                            intersection_,
                            self._radius_list[node.level - 1],
                            node.level - 1,
                            self._random_permutation_pi[j],
                        )
                        self._add_to_node_list(child_node)
                        self._add_to_level_i_sets(child_node)
                        node.add_child(child_node)

                        S -= intersection_
                        self._tree_recurs(child_node)

    def compute_tree_distance(self, u, v):
        curr_node = self.leaves[u]
        while v not in curr_node:
            curr_node = curr_node.parent
        level = curr_node.level
        return 2 ** (level + 2) - 4

    def __repr__(self, node: Node | None = None, level: int = 0) -> str:
        if node is None:
            ret = "\t" * level + repr(self.root) + "\n"
            for child in self.root.children:
                ret += self.__repr__(node=child, level=level + 1)
            return ret
        else:
            ret = "\t" * level + repr(node) + "\n"
            for child in node.children:
                ret += self.__repr__(node=child, level=level + 1)
            return ret
