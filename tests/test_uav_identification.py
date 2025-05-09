import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import problem_runner


class TestUAVIdentificationTermination:
    def test_termination_4_fully_adaptive(self):
        problem_runner.run_experiment("uav", 19, l=4, c_top=1, c_bot=4, height=10)

    def test_termination_4_non_adaptive(self):
        problem_runner.run_experiment("uav", 19, l=4, c_top=1, c_bot=4, height=10)

    def test_termination_4_2_adaptive(self):
        problem_runner.run_experiment("uav", 19, l=4, c_top=1, c_bot=4, height=10)
