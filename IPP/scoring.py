"""
Module calculates the score based on the formulas (and not groups)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import grouping
import numpy as np
import numpy.typing as npt
import sensing

if TYPE_CHECKING:
    from sensing import Observation

module_logger = logging.getLogger(__name__)


def compute_f_bar(
    partial_observation: list[Observation],
    compatible_scenario: set[int],
    sensing_matrix: npt.NDArray[np.int_],
) -> int:
    """Computes the submodular function. Which counts the number of scenarios eliminated."""
    eliminated_scenarios = sensing.find_eliminated_scenarios(
        partial_observation, compatible_scenario, sensing_matrix
    )

    return len(eliminated_scenarios)


def compute_Q_bar(compatible_scenarios: set[int]) -> int:
    """Computes the residual target, which is one less of the number of compatible scenarios."""
    return len(compatible_scenarios) - 1


def compute_information_gain(
    tour_vertices: list[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    sensing_matrix: npt.NDArray[np.int_],
) -> float:
    """
    Parameters
    ----------
    tour_vertices : list
        List of vertices in the tour.
    compatible_scenarios : set
        Set of compatible scenarios.
    probability_distribution : dict
        Probability distribution of the scenarios.
    sensing_matrix : np.array
        Sensing matrix of the problem.

    Returns
    -------
    float
        Information gain of the tour.
    """
    union_L_v = set().union(
        *[
            grouping.generate_L_v(v, compatible_scenarios, sensing_matrix)
            for v in tour_vertices
        ]
    )
    counted_scenarios = union_L_v.intersection(compatible_scenarios)
    information_gain = sum(
        probability_distribution[scenario] for scenario in counted_scenarios
    )
    return information_gain


def compute_relative_function_gain(
    tour_vertices: list[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    sensing_matrix: npt.NDArray[np.int_],
) -> float:
    """COmpute the relative function gain of the tour.

    Parameters
    ----------
    tour_vertices : list[int]
        List of locations visited by the tour
    compatible_scenarios : set[int]
        Set of compatible scenarios.
    probability_distribution : dict[int, float]
        Prior probability distribution over the scenarios.
    sensing_matrix : npt.NDArray[np.int_]
        Sensing matrix of the problem.

    Returns
    -------
    float
        Relative function gain of the tour.
    """
    relative_function_gain = 0
    for omega in compatible_scenarios:
        tour_realization_under_omega = sensing.generate_partial_realization(
            omega, tour_vertices, sensing_matrix
        )
        f_bar_omega = compute_f_bar(
            tour_realization_under_omega,
            compatible_scenarios,
            sensing_matrix,
        )
        relative_function_gain += f_bar_omega * probability_distribution[omega]
    Q_bar = compute_Q_bar(compatible_scenarios)
    relative_function_gain /= Q_bar
    return relative_function_gain
