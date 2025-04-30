import logging
from collections import defaultdict
from typing import Iterable, Optional

import numpy.typing as npt
import sensing

module_logger = logging.getLogger(__name__)


def create_large_partition(
    visited: Iterable[int],
    compatible_scenarios: set[int],
    sensing_matrix: npt.NDArray,
    delta: Optional[float] = None,
    m: Optional[int] = None,
) -> list[set[int]]:
    """Create partitions that are larger than delta*m

    Parameters
    ----------
    visited : Iterable[int]
        Set of nodes that were visited during the current iteration.
    compatible_scenarios : set[int]
        Set of scenarios that are compatible at the start of the current iteration.
        Nodes that are currently being planned are not included, as they have yet to be visited.
    sensing_matrix : npt.NDArray
        Matrix giving us the locations that can  be observed from each node.
    delta : Optional[float], optional
        Determines the stopping condition, by default None
    m : Optional[int], optional
        The number of compatible scenarios at the start of the current iteration, by default None

    Returns
    -------
    list[set[int]]
        List of large partitions.

    Raises
    ------
    ValueError
        Ensures that both delta and m are given. If both are none, allow for unfiltered partition.
    """
    # Must specify both delta and m, or leave them out for unfiltered partition.
    if delta is None and m is not None or delta is not None and m is None:
        raise ValueError("delta and m must be both None or both not None.")
    scenario_to_partial_realization = defaultdict(set)  # type: ignore

    for scenario in compatible_scenarios:
        partial_realization_of_scenario = tuple(
            sensing.generate_partial_realization(scenario, visited, sensing_matrix)
        )
        scenario_to_partial_realization[partial_realization_of_scenario].add(scenario)

    if delta is not None and m is not None:
        delta_times_m = delta * m
    else:
        delta_times_m = -1  # -1 to allow everything through

    partition = [part for part in scenario_to_partial_realization.values()]

    module_logger.debug(f"Partition: {partition}")
    large_partition = [part for part in partition if len(part) > delta_times_m]

    assert (
        set().union(*large_partition).issubset(compatible_scenarios)
    ), "Partitions do not cover all scenarios."
    assert set().intersection(*large_partition) == set(), "Partitions are not disjoint."

    return large_partition
