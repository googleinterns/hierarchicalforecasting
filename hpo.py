import sys
import os
import numpy as np
import subprocess
import pickle
import shutil
import json

def main():
    for it in range(100):
        batch_size = int(2 ** np.random.uniform(8, 11))
        l2 = 10 ** np.random.uniform(-4, 1)
        l2_slack = 10 ** np.random.uniform(-4, 1)
        node_emb = np.random.randint(4, 25)
        ep = np.random.randint(20, 30)
        lr = 10 ** np.random.uniform(-3, -1)

        hparams = {
            "batch_size": batch_size,
            "l2": l2,
            "l2_slack": l2_slack,
            "node_emb": node_emb,
            "ep": ep,
            "lr": lr
        }
        print(f'HPARAMS run {it}:', hparams)

        shutil.rmtree('logs/m5/fixed_sibling_reg/hpo/', ignore_errors=True)

        for i in range(10):
            cmd = ["python", "train.py", f"--expt=hpo/run_{i}",
                f"--random_seed={i}", "--model=fixed", "--hierarchy=sibling_reg",
                f"--batch_size={batch_size}", f"--l2_reg_weight={l2}",
                f"--l2_weight_slack={l2_slack}", f"--node_emb_dim={node_emb}",
                "--overparam=True", "--output_scaling=True", f"--train_epochs={ep}",
                f"--learning_rate={lr}"]
            with open('logs/hpo.log', 'w') as fout:
                subprocess.run(
                    cmd, check=True, stdout=fout, stderr=subprocess.STDOUT,
                    text=True
                )
        hparams['evals'] = []
        for i in range(10):
            with open(f'logs/m5/fixed_sibling_reg/hpo/run_{i}/eval.pkl', 'rb') as fin:
                eval_dict = pickle.load(fin)
            hparams['evals'].append(eval_dict)
        
        write(hparams)


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
