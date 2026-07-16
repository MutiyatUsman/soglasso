# SOGlasso: Sparse Overlapping Group Lasso

Reference implementation of the **Sparse Overlapping Group (SOG) Lasso**,
from:

> Nikhil S. Rao, Robert D. Nowak, Christopher R. Cox, Timothy T. Rogers.
> **"Classification with Sparse Overlapping Groups."** arXiv:1402.4512, 2014.
> https://arxiv.org/abs/1402.4512

## Overview

SOGlasso is a convex penalty for structured sparse feature selection when
features are organized into **possibly overlapping groups** (e.g. genes in
overlapping biological pathways, or voxels in overlapping anatomical brain
regions across subjects). It encourages solutions that are:

1. supported on a **small number of groups** (across-group sparsity), and
2. **sparse within** each selected group (within-group sparsity).

This generalizes the Lasso, the Group Lasso, and the Sparse Group Lasso —
all three are special cases of the same estimator (see `soglasso/baselines.py`
and the paper's Remarks in Sections 2.1 and 3).

The penalty (Eq. 8 in the paper):

```
h(x) = inf_{{w_G}} sum_G ( ||w_G||_2 + mu * ||w_G||_1 )    s.t.  sum_G w_G = x
```

is solved via proximal gradient descent (FISTA), using the covariate
duplication / feature replication strategy (Jacob, Obozinski & Vert, 2009)
to reduce the overlapping-group problem to a non-overlapping sparse group
lasso in an expanded space (paper, Section 3.2).

## Installation

```bash
git clone <this-repo-url>
cd soglasso
pip install -e .
# for notebooks / plotting:
pip install -e ".[tutorials]"
```

Requires Python >= 3.9. Dependencies: numpy, scipy, scikit-learn
(matplotlib only needed for the tutorial notebooks).

## Quickstart

```python
import numpy as np
from soglasso import make_sog_regression, sog_lasso

# synthetic (k, l)-group-sparse regression problem with overlapping groups
Phi, y, x_star, groups = make_sog_regression(
    n_samples=150, n_groups=30, group_size=6, shift=4,
    k_active=4, alpha=0.4, noise_std=0.1, random_state=0,
)

model = sog_lasso(groups, mu=0.5, eta1=0.5, eta2=1e-4, max_iter=400)
model.fit(Phi, y)

print("recovery error:", np.linalg.norm(model.coef_ - x_star))
```

## Tutorials

- **`notebooks/01_quickstart.ipynb`** — basic fit/predict usage, comparing
  SOGlasso's recovered support against plain Lasso and the (latent)
  Overlapping Group Lasso on a synthetic group-sparse problem.
- **`notebooks/02_reproduce_figure6a.ipynb`** — reproduces the qualitative
  result of **Figure 6(a)** / **Section 5.2** in the paper: held-out MSE of
  Lasso, Glasso, OGlasso and SOGlasso as the within-group sparsity fraction
  `alpha` is varied. SOGlasso tracks the best baseline across the full
  range, since it is the only method modeling both levels of structure.

Both notebooks run end-to-end in a few minutes on a laptop and are also
available as a standalone script: `tutorials/reproduce_figure6a.py`.

## Package structure

```
soglasso/
├── soglasso/
│   ├── proximal.py    # soft-thresholding + group soft-thresholding (Eq. 16)
│   ├── groups.py       # overlapping-group replication utilities (Sec 3.2)
│   ├── model.py         # SOGLasso estimator (FISTA solver)
│   ├── baselines.py    # Lasso / Glasso / OGlasso as special cases
│   └── data.py          # synthetic (k,l)-group-sparse data generator (Sec 5.2)
├── notebooks/            # tutorial notebooks (see above)
├── tutorials/            # standalone reproduction script + outputs
└── tests/                 # unit tests (pytest)
```

## What this reproduces (and what it doesn't)

This is a **from-scratch reference implementation** derived from the paper's
equations (no official code release accompanies arXiv:1402.4512). It
reproduces:

- The exact proximal operator of Section 3.2 / Eq. 16.
- The covariate-duplication strategy for overlapping groups.
- The qualitative comparison in Figure 6(a) (Section 5.2), on a single-task
  version of the paper's synthetic experiment, with reduced trial count for
  tractable runtime (10 trials here vs. 100 in the paper) and per-method
  hyperparameters chosen by validation-set grid search rather than the
  paper's oracle/"clairvoyant" tuning.

It does **not** attempt to reproduce the fMRI (star-plus dataset) or breast
cancer gene-pathway experiments (Sections 5.1.1 / 5.3), since those require
restricted/licensed datasets not bundled here.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Citation

```bibtex
@article{rao2014classification,
  title={Classification with Sparse Overlapping Groups},
  author={Rao, Nikhil S and Nowak, Robert D and Cox, Christopher R and Rogers, Timothy T},
  journal={arXiv preprint arXiv:1402.4512},
  year={2014}
}
```

## License

MIT
