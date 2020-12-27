import sys
import os
import numpy as np
import subprocess
import pickle
import shutil
import json


def main():
    for it in range(1000):
        batch_size = 200  # int(2 ** np.random.uniform(6, 8))
        node_emb = np.random.randint(4, 25)
        ep = np.random.randint(7, 15)
        lr = 10 ** np.random.uniform(-3, -1)
        lstm_hidden = int(2 ** np.random.uniform(2, 7))

        hparams = {
            "batch_size": batch_size,
            "node_emb": node_emb,
            "ep": ep,
            "lr": lr,
            "lstm_hidden": lstm_hidden,
        }
        print(f"HPARAMS run {it}:", hparams)

        shutil.rmtree("logs/fav/fixed/hpo/", ignore_errors=True)

        for i in range(10):
            cmd = [
                "python",
                "train.py",
                f"--expt=hpo/run_{i}",
                f"--random_seed={i}",
                "--model=fixed",
                f"--batch_size={batch_size}",
                "--emb_as_inp=True",
                f"--node_emb_dim={node_emb}",
                f"--fixed_lstm_hidden={lstm_hidden}",
                "--overparam=False",
                "--output_scaling=False",
                f"--train_epochs={ep}",
                f"--learning_rate={lr}",
            ]
            with open("logs/hpo.log", "w") as fout:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        hparams["evals"] = []
        for i in range(10):
            with open(f"logs/fav/fixed/hpo/run_{i}/eval.pkl", "rb") as fin:
                eval_dict = pickle.load(fin)
            hparams["evals"].append(eval_dict)

        write(hparams)


def write(hparams):
    hpo_file = "hpo_results_fav.pkl"
    try:
        with open(hpo_file, "rb") as fin:
            results = pickle.load(fin)
    except FileNotFoundError:
        results = []

    results.append(hparams)
    with open(hpo_file, "wb") as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    main()
