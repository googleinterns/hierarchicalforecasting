import pickle
import numpy as np
import sys
import os


def main():
    directory = sys.argv[1]
    num_runs = int(sys.argv[2])
    print('Directory:', directory)
    print('#Runs:', num_runs)

    agg = {}
    for i in range(1, num_runs+1):
        eval_path = os.path.join(directory, f'run_{i}', 'eval.pkl')
        with open(eval_path, 'rb') as fin:
            eval_dict = pickle.load(fin)
        for key in eval_dict:
            if key not in agg:
                agg[key] = []
            agg[key].append(eval_dict[key])
    
    for key in agg:
        print(key, f'{np.mean(agg[key]):.4f} +- {np.std(agg[key]):.4f}')

if __name__ == "__main__":
    main()
