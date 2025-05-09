"""
Module solves the LP relaxation of the group steiner tree problem.
"""

from __future__ import annotations

import logging
import math
import typing

import gurobipy as gp
import numpy as np
import numpy.typing as npt
from gurobipy import GRB

if typing.TYPE_CHECKING:
    from FRT_tree import Node
    from grouping import Group


module_logger = logging.getLogger(__name__)
env = gp.Env(empty=True)
env.setParam("LogToConsole", 0)
env.start()


def solve_linear_program(
    adjacency_matrix: npt.NDArray,
    group_list: list[Group],
    leaves: list[Node],
    root_node: Node,
    edge_list: list[tuple[int, int]],
    parent_dict: dict[int, int],
    children_dict: dict[int, list[int]],
) -> tuple[dict, dict, dict, gp.Model]:
    """Solves the LP relaxation of the group steiner tree problem.

    Parameters
    ----------
    adjacency_matrix : npt.NDArray
        Adjacency matrix of a tree.
    group_list : list[Group]
        List of groups.
    leaves : list[Node]
        List of leaves, where leaves[i] is the leaf that
        represents location i on the underlying grid.
    root_node : Node
        Root node of the tree. (Not the root of the grid)
    edge_list : list[tuple[int, int]]
        List of edges such that (u, v) goes from the root to the leaves.
    parent_dict : dict[int, int]
        Dictionary mapping the index of a node to the index of its parents.
    children_dict : dict[int, list[int]]
        Dictionary mapping the index of a node to the index of its children.

    Returns
    -------
    X_values : dict
        Dictionary containing the edge weigths.
    y_values : dict
        Dictionary containing the weights of each group.
    flow_valus,
        Dictionary containing the flow values.
    model : gp.Model
        Gurobipy model.
    """
    module_logger.info("Starting Linear Program")
    m = gp.Model("Group_Steiner_LP_relaxation", env=env)
    m.read("tune1.prm")
    # Keeps track of available edges on the graph

    # Flow is in the form ((u, v), i) where i is the flow type
    flow_names = [
        (edge, flow_type) for edge in edge_list for flow_type in range(len(group_list))
    ]

    # Dictionary that tells us what group the node is in.
    # {node_index: group_index}
    node_index_to_set_index_mapping = {}
    for leaf in leaves:
        node_index_to_set_index_mapping[leaf.id] = set(
            [
                group_idx
                for group_idx, group in enumerate(group_list)
                if leaf.item in group  # type: ignore
            ]
        )
    # Declaring variables
    X = m.addVars(edge_list, name="X")
    y = m.addVars(len(group_list), name="y")
    flows = m.addVars(flow_names, name="f")  # type: ignore

    m.setObjective(
        gp.quicksum([adjacency_matrix[edge] * X[edge] for edge in edge_list]),
        GRB.MINIMIZE,
    )

    for i, j in edge_list:
        par_node = parent_dict[i]
        if par_node is not None:
            # Flow must go through parent
            m.addConstr(
                X[i, j] <= X[par_node, i], f"flow_through_parent(edge[{i},{j}])"
            )

        for group_idx in range(len(group_list)):
            child_nodes = children_dict[j]
            if child_nodes:
                m.addConstr(
                    flows[(i, j), group_idx]
                    == gp.quicksum(
                        flows[(j, child_node), group_idx] for child_node in child_nodes
                    ),
                    f"flow_conservation(node={j},group={group_idx})",
                )
            else:
                if group_idx not in node_index_to_set_index_mapping.get(j, set()):
                    m.addConstr(
                        flows[(i, j), group_idx] == 0,
                        f"flow_to_leaf(leaf={j},group={group_idx})",
                    )

    # Rescale group_weights to be above 1.
    min_group_weight = min([group.group_weight for group in group_list])
    normalized_group_weights_unrounded = [
        group.group_weight / min_group_weight for group in group_list
    ]

    # Handle floating point precision errors close to 1.
    normalized_group_weights_rounded = [
        1 if math.isclose(x, 1) else x for x in normalized_group_weights_unrounded
    ]
    assert (
        normalized_group_weights_rounded == normalized_group_weights_unrounded
    ), "Rounding should not change value."
    m.addConstr(
        gp.quicksum(
            [y[i] * normalized_group_weights_rounded[i] for i in range(len(group_list))]
        )
        >= 1,
        "normalized_denominator",
    )

    m.addConstrs(
        (
            gp.quicksum([flows[e, i] for e in edge_list if e[0] == root_node.id])
            == y[i]
            for i in range(len(group_list))
        ),
        name="flow_from_root",
    )
    m.addConstrs(
        (flows[e, i] <= X[e] for e in edge_list for i in range(len(group_list))),
        name="edge_capacity",
    )

    m.optimize()
    # logger.debug(m.display())
    if __debug__:
        m.write("logs/out.lp")
    vals_X = m.getAttr("X", X)
    vals_y = m.getAttr("X", y)

    vals_flows = m.getAttr("X", flows)

    vals_X = dict(vals_X)
    vals_y = dict(vals_y)
    vals_flows = dict(vals_flows)
    module_logger.info("End LP")
    return vals_X, vals_y, vals_flows, m


def main():
    l = 8
    c_top = 1
    c_bot = 4
    height = 10

    module_logger.info(
        f"Starting identification with grid length = {l}, c_top = {c_top}, c_bot = {c_bot}, height = {height}"
    )

    # Pick a random scenario
    realized_scenario = 100

    module_logger.info(f"Realized scenario: {realized_scenario}")
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

    groups = grouping.create_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )

    gst = group_steiner_tree.GroupSteinerTree(
        locations,
        distance_matrix,
        groups,
    )

    gst.get_tour()


if __name__ == "__main__":
    import group_steiner_tree
    import grouping
    import initialization

    main()
