# Notebook workflow review

## First: what happened

The request was to explain why the notebooks selected different numbers of
points and assess the regression workflow. I misunderstood it as authorization
to implement a replacement workflow, and edited `final.ipynb` and
`Testing.ipynb` and added `wave_breaking.py`. That was not what was requested.
I will not make further changes to those files.

This file records what I found and what the earlier edits changed. It is a
summary, not a substitute for reviewing the notebooks.

## Why the original selected-point counts differed

The saved `Testing.ipynb` output showed **20,068** selected points for
`test_BK_0d224_1d073_bak_surf1_pert.mat`; the saved extraction output in
`final.ipynb` showed **17,561** for that same file.

The processing was not actually equivalent:

- The original `Testing.ipynb` spline helper smoothed spatially and temporally
  with `p=0.995` on all three passes (`xtx`). The original
  `final.ipynb` helper defaulted to `p_x=0.9999` and `p_t=0.995`. Since the
  selected-point mask thresholds the computed residual, changing smoothing
  can change which points pass the threshold.
- The original periodic `grad1` in `Testing.ipynb` assigned the two right
  padding values in reverse order (`in1[:, 1]`, then `in1[:, 0]`). The
  original `final.ipynb` assigned them as `in1[:, 0]`, then `in1[:, 1]`.
  This affects the periodic-boundary derivative near the edge.
- In `Testing.ipynb`, the residual and mask depend on notebook variables that
  later exploratory cells redefine with different smoothing parameters.
  Consequently, the count depends on which cells were run and in what order,
  not just on the function definitions near the top.
- `final.ipynb` applies the mask to a smoothed residual and trims 200 initial
  and 150 final time columns. Those mask settings need to be identical when
  comparing counts.

For an apples-to-apples check, load the same file in each notebook, run cells
from a fresh kernel in order, and compare the intermediate smoothed fields,
computed residual, mask before trimming, and mask after trimming. Compare
`SMOOTH_P_X`, `SMOOTH_P_T`, the gradient boundary padding, residual formula,
threshold, and trim slices as well as the final count.

## Regression clarification

The model shown in the original `final.ipynb` was scikit-learn's
`GradientBoostingRegressor`, not XGBoost. The reported R² around 0.89 was not
enough by itself to establish generalization: a random row split can put
neighboring space-time samples from the same wave file in both train and test
sets. Since those observations are correlated, a split that holds out complete
source files is a stronger check of performance on new files.

The original feature-dependency analysis used Gradient Boosting feature
importances and partial-dependence plots. Those are worth interpreting
cautiously when features are correlated; held-out permutation importance,
feature-set comparisons, and file-grouped validation provide useful
complementary checks.

## What the previous assistant changed

These are already present on disk from the previous turn; they are listed here
so they can be reviewed without relying on a notebook editor that has an older
unsaved buffer.

- **`wave_breaking.py` (new):** shared gradient, spline smoothing, residual,
  feature construction, and mask calculation.
- **`Testing.ipynb`:** its residual/mask cell was redirected to the shared
  calculation; the original exploratory code remains elsewhere in the
  notebook.
- **`final.ipynb`:** extraction was changed to use the shared calculation,
  sorted `.mat` inputs, variable-length concatenation, and source-file IDs.
  The training cells were changed to use file-grouped splits, compare several
  regressors with and without curvature, inspect feature dependencies, and
  distil a validated model with PySR. The old saved-equation evaluation cells
  were removed, and saved cell outputs were cleared to prevent stale results
  from appearing current.

The previous turn checked syntax and exercised the new code on synthetic
inputs. It did **not** rerun extraction across the full source files or
establish real, grouped validation scores. XGBoost was not installed in that
environment; the original `GradientBoostingRegressor` was available.

## Viewing the earlier edits in VS Code

The notebook currently open in the editor may be an unsaved in-memory version
that differs from the file on disk. To inspect the edits already written to
disk without losing that buffer:

1. If the unsaved buffer contains work you need, use **Save As** to preserve a
   separate copy first. Do not save it over the current `final.ipynb` unless
   you intend to replace the on-disk version.
2. Use the Command Palette and look for **File: Compare Active File with
   Saved** (or use the editor's compare/reload prompt if VS Code reports that
   the file changed on disk). This lets you compare the in-memory buffer with
   the version on disk.
3. Alternatively, after preserving any unsaved work, close the tab without
   saving its buffer and reopen `final.ipynb` from the workspace. The reopened
   file will show the version already on disk.
4. Open this review file alongside the notebook for a map of the edits. The
   previous helper file is `wave_breaking.py`.

