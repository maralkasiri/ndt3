# Session occlusion

These experiments evaluate whether or not we still experience gains from multi-session scaling after pretraining.
Vs. Cursor, this is long timescale (O(100d)), NHP data. Long timescale is statistically different shift than close timescale.

This multi-session transfer is itself likely occluded by in-session calibration. Like in other experiments, we maintain a small
amount of in-session calibration.

We justify only using 45m/350m at 2kh because
- high likelihood / empirics of seeing different trends at different capacities
- performance is approximately similar for 2kh / 200h in the tasks of interest (2D + click and RTT).


Reasons that sessions would be occluded:
- Given high structure / task simplicity, it's possible that once the model just "gets it" in the new day, there's little to be gained from scaling data further generically. e.g. NoMAD has perfect performance on novel days from one day of data, so the latent structure is sufficient.

Why sessions might not be occluded:
- From scaling trends in NDT2, JY presumes yes, multi-session data is _very_ close to in-session scaling. Unlikely that broad data could occlude this.

Note that while most models are joint tuned 9x (3 seed x 3 lr) scratch_base -> scratch transfer does this 1x for base and 9x for transfer (rather than a nominally more fair 3x3, which would require more implementation)