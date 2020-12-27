import pickle
import numpy as np
import sys
import os
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def main():
    directory = sys.argv[1]
    num_runs = int(sys.argv[2])
    print("Directory:", directory)
    print("#Runs:", num_runs)

    agg = {}
    for i in range(1, num_runs + 1):
        eval_path = os.path.join(directory, f"run_{i}", "eval.pkl")
        with open(eval_path, "rb") as fin:
            eval_dict = pickle.load(fin)
        for key in eval_dict:
            if key not in agg:
                agg[key] = []
            agg[key].append(eval_dict[key])

    for key in agg:
        bsr = bs.bootstrap(
            np.asarray(agg[key]), stat_func=bs_stats.mean, alpha=0.05
        )  # 95% confidence interval
        print(
            key,
            f"Mean: {np.mean(agg[key]):.4f}\tStd: {np.std(agg[key]):.4f}"
            f"\tBootstrap CI: ({bsr.lower_bound:.4f}, {bsr.upper_bound:.4f})",
        )


if __name__ == "__main__":
    main()
