### Top 10 Best Runs

| Rank | Model Configuration | Epochs | LR | $T$ | Optimizer | **FID Score** |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | `run_lr0.0002_T750_adamw_linear` | **125** | `2e-4` | **750** | **AdamW** | **60.75** |
| **2** | `run_lr0.0005_T750_adam_linear` | **200** | `5e-4` | **750** | **Adam** | **67.91** |
| **3** | `run_lr0.0002_T750_adam_linear` | **125** | `2e-4` | **750** | **Adam** | **68.72** |
| **4** | `run_lr0.0001_T750_adamw_linear` | **200** | `1e-4` | **750** | **AdamW** | **73.12** |
| **5** | `run_lr0.0005_T750_adam_linear` | **175** | `5e-4` | **750** | **Adam** | **76.50** |
| **6** | `run_lr0.0002_T750_adamw_linear` | **200** | `2e-4` | **750** | **AdamW** | **80.21** |
| **7** | `run_lr0.0002_T750_adamw_linear` | **150** | `2e-4` | **750** | **AdamW** | **82.14** |
| **8** | `run_lr0.0001_T750_adam_linear` | **200** | `1e-4` | **750** | **Adam** | **85.65** |
| **9** | `run_lr0.0005_T1000_adamw_linear` | **200** | `5e-4` | **1000** | **AdamW** | **89.04** |
| **10** | `run_lr0.0005_T750_adamw_linear` | **200** | `5e-4` | **750** | **AdamW** | **90.17** |

---

1. T=750 dominates T=1000
* T=750 Average FID: `103.80`
* T=1000 Average FID: `121.32`

2. Epoch 125
* The single best model (`lr=2e-4`, `T=750`, `AdamW`) peaked at **Epoch 125** with **FID 60.75**, before its FID rose to 82.14 at 150 epochs and 80.21 at 200 epochs.


3. Learning Rate & Optimizer Stability
* `AdamW` continues to edge out `Adam` on average (`108.89` vs `117.23`), but `Adam` with `lr=5e-4` or `2e-4` achieved very strong runner-up scores (67.91 and 68.72).
* Learning rates in the `2e-4` to `5e-4` window hit the global minimum faster than `1e-4`.