# Not upkept for a while...
import json
import os
from omegaconf import OmegaConf
from context_general_bci.config import DatasetConfig
from context_general_bci.config.presets import ScaleHistoryDatasetConfig
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.contexts.context_registry import context_registry
from context_general_bci.contexts.context_info import ContextInfo
from context_general_bci.dataset import SpikingDataset, preprocess_path
from multiprocessing import Pool

# Your existing code for context querying and config setup
# contexts = context_registry.query(alias='schwartz_Nigel.*')
# contexts = context_registry.query(alias='schwartz_Rocky.*542.*')
contexts = context_registry.query(alias='schwartz_Rocky.*')
cfg: DatasetConfig = OmegaConf.create(ScaleHistoryDatasetConfig())
cfg.pack_dense= True
if not isinstance(contexts, list):
    contexts = [contexts]
# assert isinstance(contexts, list)
print(len(contexts))

# Your existing function definitions (preproc_version and preprocess_single_session)

def preproc_version(cfg: DatasetConfig, task: ExperimentalTask):
    version = {
        'max_trial_length': cfg.max_trial_length,  # defunct
        'bin_size_ms': cfg.bin_size_ms,
        'tokenize_covariates': cfg.tokenize_covariates,
        'return_horizon_s': cfg.return_horizon_s,
    }
    task_cfg = getattr(cfg, task.value)
    task_dict = OmegaConf.to_container(task_cfg, resolve=True)
    for k, v in task_dict.items():
        version[k] = v
    return version

def preprocess_single_session(ctx: ContextInfo):
    session_path = ctx.datapath
    print(f"Preprocessing {session_path}")
    hash_dir = preprocess_path(cfg, session_path, override_preprocess_path=False)
    os.makedirs(hash_dir, exist_ok=True)
    if (hash_dir / 'preprocess_version.json').exists():
        print(f"\tAlready proc-ed, skipping {session_path}")
        return
    meta = ctx.load(cfg, hash_dir)
    meta.to_csv(hash_dir / 'meta.csv')
    with open(hash_dir / 'preprocess_version.json', 'w') as f:
        json.dump(preproc_version(cfg, ctx.task), f)
    print(f"\tFinished {session_path}: {len(meta)} trials")

# Wrapper function for multiprocessing
def preprocess_sessions(contexts):
    # for ctx in contexts:
        # preprocess_single_session(ctx)

    # exit(0)
    # cpu_cap = 96
    cpu_cap = 32
    from tqdm.contrib.concurrent import process_map
    # process_map(preprocess_single_session, contexts[::-1], max_workers=min(cpu_cap, os.cpu_count()))
    process_map(preprocess_single_session, contexts, max_workers=min(cpu_cap, os.cpu_count()))
    # with Pool(processes=min(cpu_cap, os.cpu_count())) as pool:  # Adjust the number of processes as needed
        # r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
        # pool.map(preprocess_single_session, contexts)

# Main execution
if __name__ == "__main__":
    preprocess_sessions(contexts)
