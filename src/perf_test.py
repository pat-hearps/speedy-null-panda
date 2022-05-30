import json
from pathlib import Path
import timeit

import numpy as np
import pandas as pd

import src.algorithms as alg

functions = {
    "index": (alg.index_find_max_consec_nulls, False),
    "groupby": (alg.pdgroupby_find_max_consec_nulls, True),
    "class": (alg.class_find_max_consec_nulls, True),
    "basic": (alg.basic_find_max_consec_nulls, True),
    "number": (alg.numba_find_max_consec_nulls, True),
}


def main():
    """Run the main program"""
    file = Path(__file__).parent / "results.json"
    file.touch()
    number = 1
    for nrows in [10, 100, 1000]:
        for ncols in [10, 100, 1000]:
            for algorithm in functions:
                total_time = time_algorithm(algorithm, nrows, ncols, number)
                avg_time = total_time / number
                result = {
                    "algorithm": algorithm,
                    "nrows": nrows,
                    "ncols": ncols,
                    "avg_time": avg_time,
                }
                print(result)
                append_results(result, file)


def time_algorithm(algorithm: str, nrows: int, ncols: int, number: int = 1) -> float:
    """Time the algorithm function"""
    func, pre_null = functions[algorithm]
    df = make_df_random(nrows=nrows, ncols=ncols)

    if pre_null:
        df = df.isnull()

    return timeit.timeit(lambda: df.apply(func, axis=1), number=1)


def make_df_random(nrows: int, ncols: int, seed: int = 987654) -> pd.DataFrame:
    """Make dataframe of random integers from same seed with variable size.
    All cells with values >= 3 are set to np.NaN (null)"""
    RS = np.random.RandomState(seed=seed)
    data = RS.randint(low=0, high=10, size=(nrows, ncols))
    df = pd.DataFrame(data)
    df = df[df >= 3]  # fills all values less than 3 with np.NaN
    return df


def append_results(results: dict, file: Path) -> None:
    with file.open("a") as out:
        out.write(f"{json.dumps(results)},\n")


if __name__ == "__main__":
    main()
