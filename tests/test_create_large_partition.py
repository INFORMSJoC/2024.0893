import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import initialization
import k_rounds_adaptive_iteration
import uav_initializer


class TestCreatePartition:
    n = 8
    adjacency_matrix = uav_initializer.create_uav_adjaceny_matrix(n, 1, 4, 10)
    dist_matrix, predecessors = initialization.create_distance_matrix(adjacency_matrix)
    locations = set(range(2 * n**2 + 1))
    compatible_scenario_full = set(range(n**2, 2 * n**2))
    sensing_matrix = uav_initializer.create_uav_sensing_matrix(n)

    def test_partition(self):
        partition = k_rounds_adaptive_iteration.create_large_partition(
            [0, 1],
            self.compatible_scenario_full,
            self.sensing_matrix,
        )
        assert len(partition) == 3
        assert partition == [
            {64, 65, 72, 73},
            {66, 74},
            set(range(64, 128)) - {64, 65, 66, 72, 73, 74},
        ]
        partition = k_rounds_adaptive_iteration.create_large_partition(
            range(0, 64),
            self.compatible_scenario_full,
            self.sensing_matrix,
        )
        assert len(partition) == 64
        assert partition == [{i} for i in range(64, 128)]
