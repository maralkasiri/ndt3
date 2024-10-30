#%%
from typing import List 
from pathlib import Path
import subprocess 
import pandas as pd

def merge_eval_csvs(local_csv_path: Path, remote_csv_paths: List[str], unique_columns: List[str], eval_columns: List[str]):
    r"""
        Assuming that the remote paths are accessible via ssh config
        Remote paths should include hostname and full path
    
        Heck. In general this is not good because data is usually either living behind inconvneient MFA or VPNs. Neither obviously scriptable.
    """
    # Copy each file in the remote paths to a temp local path
    local_dir = local_csv_path.parent
    all_dfs = []
    all_dfs.append(pd.read_csv(local_csv_path))
    for remote_csv_path in remote_csv_paths:
        host, remote_path = remote_csv_path.split(':')
        print(remote_csv_)
        status = subprocess.run(f"scp {remote_csv_path} {local_dir / f'{host}_{local_csv_path.name}'}", shell=True)
        if status.returncode != 0:
            print(f"Failed to copy {remote_csv_path} to {local_dir / f'{host}_{local_csv_path.name}'}")
            continue
        all_dfs.append(pd.read_csv(local_dir / f'{host}_{local_csv_path.name}'))
    # Filter out duplicates, based on unique columns. Prioritize nonzero.
    df = pd.concat(all_dfs)
    df.sort_values(by=eval_columns, inplace=True)
    df.drop_duplicates(subset=unique_columns, keep='first', inplace=True)
    return df

test = merge_eval_csvs(
    Path('~/projects/ndt3/data/eval_gen/eval_vel_co.csv'),
    ['mind:projects/ndt3/data/eval_gen/eval_vel_co.csv'],
    ['variant', 'experiment', 'held_in', 'held_out'],
    ['eval_r2']
)