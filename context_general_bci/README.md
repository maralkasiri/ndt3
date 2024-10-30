# Brief overview of codebase layout.

Vital top level files:
- `model`: Defines NDT3 `pytorch-lightning` module.
- `task_io`: Readin and readout layers for different modalities.
- `dataset`: Pytorch dataset with hooks into automatic preprocessing. Used in training and inference.
- `components`: Core Transformer block with connections to `flash-attn`. May connect to `batch_rotary`.
- `analyze_utils`: Catch-all library for different analysis tools.

Peripheral top-level files:
- `augmentations`: Augmentations computed on-the-fly during dataloading.
- `callbacks`: Alternative tuning strategies.
- `ndt3_falcon`: Wrapper for FALCON challenge.
- `ndt3_slim`: Trimmed module for inference.
- `streaming_utils/data_utils`: Formatting streamed data into batch data for model compatability. Used for realtime control.
These are safe to ignore, usually developed and abandoned after unpromising results.

Modules:
- `subjects`: Holds in-code metadata for subjects from different datasets. This was designed to support context embeddings in NDT2. Since context embeddings were dropped in NDT3, this can be deprecated with some work. The metadata is still useful, however, for analysis.
- `tasks`: Defines preprocessing on raw datasets from different tasks. Also contains data viewers.
- `contexts`: Manages directory structure of different datasets visible to the dataloader. Indexes raw datafiles and adds metadata for preprocessing.
- `utils`: Relatively standalone utilities.
- `plotting`: Plotting utilities.
- `inference`: Minor module that should be regrouped into `utils`. Mainly for converting wandb info into model and data objects.
- `external`: External copy-pasted code.
- `rtndt/rl`: Early explorations and integrations into realtime BCI infrastructure.
