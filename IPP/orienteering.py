import group_steiner_tree
import numpy.typing as npt
from grouping import Group


def orienteer(
    groups: list[Group],
    locations: set[int],
    distance_matrix: npt.NDArray,
    number_of_trials: int = 1,
) -> list[int]:
    gst = group_steiner_tree.GroupSteinerTree(
        locations,
        distance_matrix,
        groups,
    )

    gst.get_tour()
    _, tour_on_grid = gst.create_eulerian_tour()
    return tour_on_grid
