import logging

import initialization
from k_rounds_adaptive_iteration import partial_covering_planning
import numpy as np
import pandas as pd
from fully_adaptive_identification import fully_adaptive_identification
from initialize_logger import initialize_logger
from k_rounds_adaptive_identification import k_rounds_adaptive_identification
import time

logger = logging.getLogger(__name__)


def run_experiment(**kwargs):
    np.random.seed(0)
    problem = kwargs.get("problem")
    number_of_rounds = kwargs.get("number_of_rounds")
    (
        n,
        adjacency_matrix,
        distance_matrix,
        sensing_matrix,
    ) = initialization.initialize_data(**kwargs)
    if problem == "road":
        kwargs["num_scenarios"] = sensing_matrix.shape[1]
    (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    ) = initialization.initialize_variables(**kwargs)
    first_tour_planning_time = None
    first_tour = None
    if number_of_rounds is not None:
        delta = len(compatible_scenarios) ** (-1 / number_of_rounds)
        logger.info("Planning first tour.")
        first_tour_starting_time = time.process_time_ns()
        first_tour = partial_covering_planning(
            visited,
            compatible_scenarios,
            probability_distribution,
            locations,
            delta,
            distance_matrix,
            sensing_matrix,
        )
        first_tour_ending_time = time.process_time_ns()
        first_tour_planning_time = first_tour_ending_time - first_tour_starting_time
        logger.info(f"First tour: {first_tour}")

    possible_scenarios = sorted(list(compatible_scenarios))
    row_names = [
        "visited",
        "partial_realization",
        "item_",
        "grand_tour_cost",
        "iteration",
        "time_taken",
        "planning_time_record",
    ]
    assert len(possible_scenarios) > 0
    result_list = []
    for realized_scenario in possible_scenarios:
        logger.info(f"Starting scenario {realized_scenario}")
        (
            visited,
            partial_realization,
            locations,
            compatible_scenarios,
            probability_distribution,
        ) = initialization.initialize_variables(**kwargs)
        if number_of_rounds is None:
            results = fully_adaptive_identification(
                n,
                visited,
                partial_realization,
                locations,
                compatible_scenarios,
                probability_distribution,
                realized_scenario,
                distance_matrix,
                sensing_matrix,
            )
        else:
            assert number_of_rounds is not None
            results = k_rounds_adaptive_identification(
                n,
                number_of_rounds,
                visited,
                partial_realization,
                locations,
                compatible_scenarios,
                probability_distribution,
                realized_scenario,
                distance_matrix,
                sensing_matrix,
                terminate_tour_early=False,
                first_tour=first_tour,
            )

        result_list.append(dict(zip(row_names, results)))
    results = pd.DataFrame(result_list, columns=row_names)
    if number_of_rounds is not None:
        results["first_tour_planning_time"] = first_tour_planning_time
    results["problem_arguments"] = str(kwargs)
    print(results)
    return results


if __name__ == "__main__":
    initialize_logger(logger)
    run_experiment("uav", l=8, c_top=1, c_bot=4, height=10, number_of_rounds=10)
