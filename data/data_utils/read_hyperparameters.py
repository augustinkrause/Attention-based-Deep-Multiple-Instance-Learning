import argparse
import os
from typing import List
import json
from pathlib import Path


def main():
    out_path = get_args().out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    datasets=['MUSK1', 'MUSK2', 'ELEPHANT', 'TIGER', 'FOX', 'MNIST']
    mil_types=['embedding_based', 'instance_based']
    pooling_types=['max', 'mean', 'attention', 'gated_attention']
    for ds in datasets:
        for mil in mil_types:
            for pool in pooling_types:
                if mil == "instance_based" and (pool == "attention" or pool == "gated_attention"): break
                path = get_path(ds, mil, pool)
                n = 10 if ds != "MNIST" else 417
                lines = read_n_last_lines(path, n)
                print(f"Finding hyperparamters for model: {ds}_{mil}_{pool}")
                for line in lines:
                    if line.startswith("CV found the following"):
                        line = line.split("combination: ")[-1].replace("'", '"')
                        hp = json.loads(line)
                        with open(os.path.join(out_path, f"{ds}_{mil}_{pool}.json"), 'w') as f:
                            json.dump(hp, f)
                        print(f"Stored hyperparamters for model {ds}_{mil}_{pool} in {out_path}")
                        break
    print(f"Finished!")




def get_path(dataset, mil_type, pooling_type) -> str:
    return os.path.join("./data", "logs", "cv_rename", f"{dataset}_{mil_type}_{pooling_type}.out")

def read_n_to_last_line(filename, n = 1) -> str:
    """Efficiently returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    with open(filename, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)    
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line

def read_n_last_lines(filename, n = 1) -> List[str]:
    """Returns the nth before last line of a file (n=1 gives last line)"""
    r = range(n,0,-1)
    strs = []
    for i in r:
        strs.append( read_n_to_last_line(filename, n=i) )
    return strs

def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()