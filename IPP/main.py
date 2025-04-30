import argparse
import json
import logging
import os
import time

import initialize_logger
import numpy as np
import problem_runner
from sensing import Observation

dir_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "data")
results_path = os.path.join(dir_path, "results")


def main(**kwargs):
    logger.info(f"Ran with arguments: {kwargs}")
    logger.info("Running main function")
    problem_index = kwargs.pop("problem_index")
    with open(f"{data_path}/instances.json") as f:
        problem_list = json.load(f)
        problem_arguments = problem_list["problems"][problem_index]
        print(problem_arguments)

    problem_arguments["number_of_rounds"] = kwargs.pop("number_of_rounds")

    results = problem_runner.run_experiment(**problem_arguments)

    file_name = f"{results_path}/{problem_arguments['name']}_{problem_arguments['number_of_rounds']}"
    results.to_csv(file_name + ".csv")
    results.to_pickle(file_name + ".pkl")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the identification algorithm.",
        epilog="python main.py uav -l 3",
    )

    parser.add_argument(
        "-k",
        "--number_of_rounds",
        type=int,
        help="The number of rounds to run the algorithm.",
    )
    parser.add_argument(
        "-p",
        "--problem_index",
        type=int,
        help="The problem number to run.",
    )
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    arguments = parse_arguments()
    logger = logging.getLogger()
    initialize_logger.initialize_logger(logger)
    main(**arguments)
