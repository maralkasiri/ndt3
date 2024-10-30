# V6
Post-ICLR, we now will add two additional experiments to further methodological rigor from the point of model preparation.
1. Explore some intuitively important pretraining parameters to see if we can get better performance.
2. Compare neural reconstruction only objective and its sensitivity to neural permutation invariance vs full joint objective.
    - We restrict permutation invariance exploration to neural data here because while neural reconstruction seems totally feasible through in-context learning, covariate in-context learning would require providing covariates for our evaluation, which is likely too large a shift from prior experiments.
    - Neural reconstruction pretraining also lets us look at whether we see scaling from neural representation learning or covariate representation learning, since NDT2 illustrates scaling in the former alone but NDT3 is confounded.


Critical setup changes:
- Changed evaluation set - more diverse, less volume per task.

1. HP changes: Experiments are documented on https://wandb.ai/joelye9/ndt3/reports/V6-Pretraining-HPs--Vmlldzo5NTg5NDU5.
- [x] Prefix blocking and weight upscaling. Adopted. Moderately useful in 1kh as well.
- [x] Sweep: Mask minimum and prefix ratio. No effect, keeping defaults of prefix ratio 0.9 and mask min 0.
- [x] Shape: Compare standard shape from literature, 512-12-4 (51M), vs v5 1024-6-1 (45M).
    - 512-12-4 yielded slightly worse performance on min. Keeping existing shape.
