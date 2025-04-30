import pandas as pd
import os
import json

dir_path = os.path.dirname(os.path.dirname(__file__))
result_path = os.path.join(dir_path, "results")


def result_tabulation():
    files = os.listdir(result_path)
    results = []
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(result_path, file)
            with open(file_path) as f:
                result = json.load(f)
                results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_path, "results.csv"))


if __name__ == "__main__":
    result_tabulation()
