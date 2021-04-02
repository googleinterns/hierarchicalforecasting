import sys
import os
import numpy as np
import subprocess
import pickle
import shutil
import json

def main():
    dataset = 'favorita'
    
    start = 500
    for it in range(500):
        try:
            with open(f'logs/{dataset}/hpo_local/metrics_{it}.pkl', 'rb') as fin:
                df = pickle.load(fin)
        except FileNotFoundError:
            start = it
            break

    print('Start iter', start)

    for it in range(start, 500):
        batch_size = 500
        l2_act = 10 ** np.random.uniform(-10, -5)
        l2_emb = 10 ** np.random.uniform(-5, 0)
        l2_loc = 10 ** np.random.uniform(-10, -5)
        node_emb = 8
        ep = 40
        lr = 0.003
        lstm_hidden = 20

        hparams = {
            "batch_size": batch_size,
            "l2_act": l2_act,
            "l2_emb": l2_emb,
            "l2_loc": l2_loc,
            "node_emb": 8,
            "ep": ep,
            "lr": lr,
            "lstm_hidden": lstm_hidden,
            "dataset": dataset
        }
        print(f'HPARAMS run {it}:', hparams)

        shutil.rmtree(f'logs/{dataset}/hpo_local/run/', ignore_errors=True)
        
        cmd = ["python", "train.py", f"--expt=hpo_local/run/",
            f"--random_seed=1", f"--dataset={dataset}",
            f"--act_reg_weight={l2_act}", f"--emb_reg_weight={l2_emb}",
            "--local_model=True", f"--loc_reg={l2_loc}",
            f"--node_emb_dim={node_emb}", f"--fixed_lstm_hidden={lstm_hidden}",
            f"--batch_size={batch_size}", f"--train_epochs={ep}",
            f"--learning_rate={lr}", f"--num_changes=6", "--patience=10",
            "--hist_len=28", "--train_pred=28", "--test_pred=7", "--val_windows=5",
            "--test_windows=5"]
        with open('logs/hpo.log', 'w') as fout:
            subprocess.run(
                cmd, check=True, stdout=fout, stderr=subprocess.STDOUT,
                text=True
            )
        
        shutil.copyfile(
            f'logs/{dataset}/hpo_local/run/metrics.pkl',
            f'logs/{dataset}/hpo_local/metrics_{it}.pkl'
        )
        with open(f'logs/{dataset}/hpo_local/params_{it}.pkl', 'wb') as fout:
            pickle.dump(cmd, fout)


def write(hparams):
    hpo_file = 'hpo_results.pkl'
    try:
        with open(hpo_file, 'rb') as fin:
            results = pickle.load(fin)
    except FileNotFoundError:
        results = []
    
    results.append(hparams)
    with open(hpo_file, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    main()
