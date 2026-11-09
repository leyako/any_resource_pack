The results from this evaluation run allowed us to realize some things:
- The cosine scheduler needs to have dedicated learned reverse process variances, we won't be able to
implement it for the time of the project
- We keeps from these 48 models (only 24 without the cosine scheduling) only the 10 bests and we will be doing
more experiments on those, especially more epochs.

The grid_search and training will also save the optimizer's state_dict so if for example we stop at 75 epochs,
we can resume from 75 epochs with the correct state_dict for the optimizer and go further to more than 100 epochs.

The top 10 models retained are:
- run_042: 75 epochs, LR 5e-4, T=750, AdamW (FID: 90.11)
- run_026: 75 epochs, LR 1e-4, T=750, AdamW (FID: 99.23)
- run_032: 75 epochs, LR 2e-4, T=750, Adam (FID: 102.07)
- run_046: 75 epochs, LR 5e-4, T=1000, AdamW (FID: 109.83)
- run_010: 50 epochs, LR 2e-4, T=750, AdamW (FID: 111.32)
- run_018: 50 epochs, LR 5e-4, T=750, AdamW (FID: 120.01)
- run_006: 50 epochs, LR 1e-4, T=1000, AdamW (FID: 120.24)
- run_040: 75 epochs, LR 5e-4, T=750, Adam (FID: 122.42)
- run_022: 50 epochs, LR 5e-4, T=1000, AdamW (FID: 123.72)
- run_024: 75 epochs, LR 1e-4, T=750, Adam (FID: 127.68)
- run_034: 75 epochs, LR 2e-4, T=750, AdamW (FID: 128.71)
- run_014: 50 epochs, LR 2e-4, T=1000, AdamW (FID: 129.38)

We will now only do experiments on those configurations for the grayscale part.