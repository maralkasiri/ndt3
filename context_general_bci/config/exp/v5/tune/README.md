# Running FT-ing sweeps

Inherit ckpts while HP-sweeping for several different experimental settings on different clusters, can be challenging. Base commands here might be:

- Launch several slurm jobs, parallelizing HP sweep: `python run.py +exp/v5/tune/falcon_h1=base_45m_200h_100 serial_run=False cancel_if_run_exists=False`
- Launch HP sweep in serial: `sbatch ./slurm/nersc_24h.sh +exp/v5/tune/falcon_h1=base_45m_200h_100 serial_run=True cancel_if_run_exists=False`
