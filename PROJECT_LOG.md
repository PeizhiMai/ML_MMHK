# Project Log: ML Study of the 4x4-MMHK Model

## 2026-04-16 | Project kickoff

### Core goal
Use machine learning methods to study the physics of the 4x4-MMHK model, starting from the spectral function at half-filling along special momentum-space directions.

This first round is designed to answer a simple question:
What physical structure can ML recover directly from spectral data when the interaction parameters `(U,V)` are varied across a broad but organized grid?

### Round 1 scope
- System: 4x4-MMHK model
- Filling: half-filling
- Observable: spectral function
- Momentum path: special directions `Gamma -> M -> X -> Gamma`
- Control parameters: `U` and `V`
- First-round parameter count: 71 sets of `(U,V)`

The current `(U,V)` sweep is defined in `create_code/UVset.txt`.

### Data-generation setup
The current workflow already reflects this first round:
- Momentum path is listed in `create_code/kxkyset.txt`
- `(U,V)` pairs are listed in `create_code/UVset.txt`
- Job generation for spectral calculations is handled by `create_code/job_submit/generate_jobs_G_UV.sh`

The generated files appear to include both particle and hole Green-function outputs:
- `*_Gp.dat`
- `*_Gh.dat`

These should form the raw inputs for the first ML-ready dataset after parsing, alignment, normalization, and feature packaging.

### Initial physics questions
This round should help us test whether the spectral function contains enough information to learn:
- which regions of `(U,V)` produce qualitatively distinct spectral shapes
- whether ML can separate metallic-like, weakly gapped, and strongly gapped behavior
- whether changes in `V` at fixed `U` produce identifiable signatures along the special momentum path
- whether there are clusters, boundaries, or crossover lines in parameter space visible directly in spectral data
- whether a low-dimensional latent representation organizes spectra by interaction strength in a physically meaningful way

### Initial ML questions
Before aiming for a complicated model, it is worth checking a sequence of increasingly ambitious tasks:

1. Unsupervised structure discovery
   - PCA, UMAP, t-SNE, clustering
   - Goal: see whether spectra self-organize by `U`, `V`, gap size, bandwidth changes, or line-shape features

2. Parameter inference
   - Train regressors or conditional models to predict `U`, `V`, or both from spectral inputs
   - Goal: measure how invertible the mapping from interaction parameters to spectral signatures is

3. Regime classification
   - Build labels from physics-inspired criteria or clustering
   - Goal: detect emergent phases/crossovers without hand-inspecting every spectrum

4. Representation learning
   - Autoencoders or contrastive embeddings
   - Goal: learn compressed coordinates that may expose physically meaningful manifolds

### Working hypotheses
- Increasing `U` should progressively reshape spectral weight and may enhance gap-like or incoherent features.
- Changing `V` at fixed `U` may shift spectral weight in a way that is systematic enough for ML to detect.
- The special-path spectra may already contain enough information to identify broad interaction regimes, even before using full Brillouin-zone data.
- If the latent space is smooth in `(U,V)`, ML may reveal crossover structure before we define explicit order parameters.

### Practical dataset plan
For each `(U,V)` point, the first ML dataset should ideally store:
- metadata: `U`, `V`, filling, lattice size, momentum index, path segment
- spectral arrays sampled on a common frequency grid
- optional derived features: peak positions, gap estimates, spectral moments, integrated weights
- provenance: source files, script version, generation date, preprocessing notes

### Immediate next steps
1. Generate and collect the 71 `(U,V)` spectral datasets at half-filling.
2. Convert raw `Gp/Gh` outputs into a single standardized spectral representation.
3. Build a clean ML table/tensor with shared axes and metadata.
4. Make baseline visual diagnostics before training:
   - spectrum overlays
   - heatmaps along the path
   - simple gap or peak-location summaries versus `(U,V)`
5. Start with unsupervised learning and simple regressors before moving to deeper models.

### What success looks like in round 1
Round 1 is successful if we can do at least one of the following convincingly:
- identify clear spectral families across the 71 parameter points
- predict `U` and/or `V` with nontrivial accuracy from spectral inputs
- find a low-dimensional embedding that correlates with interpretable physics
- detect candidate crossover boundaries worth following up with more targeted calculations

### Risks and limitations
- Spectra along only one special path may miss structure present elsewhere in momentum space.
- With 71 parameter points, model complexity must stay modest to avoid overfitting.
- Any learned pattern must be checked against simple physics observables and not treated as meaningful by default.
- Preprocessing choices may strongly affect apparent ML structure, so provenance tracking matters.

### Experiment record template
Use the following block for each major ML run or preprocessing revision.

#### Entry template
- Date:
- Dataset version:
- Input representation:
- Model / method:
- Target:
- Train/validation split:
- Metrics:
- Main observation:
- Physics interpretation:
- Next action:

### Notes
- Start simple.
- Prefer interpretable baselines first.
- Let the data tell us what is learnable before scaling up model complexity.

## 2026-04-16 | Job progress update

### Execution status
- The required job scripts have been prepared.
- The jobs for the first-round spectral dataset have been submitted.
- Current completion: about 15%
- Expected completion: by Monday next week

### Current interpretation
The data-generation stage is now actively running. If progress remains steady, the first full `(U,V)` sweep should soon be ready for preprocessing and baseline exploratory analysis.

### Planned actions once jobs finish
1. Check output completeness across all 71 `(U,V)` points.
2. Verify file integrity and consistency of the generated `Gp/Gh` data.
3. Convert the raw outputs into a standardized ML-ready representation.
4. Run first-look plots to see whether obvious spectral trends already appear across parameter space.

## 2026-04-26 | Dataset complete and ready for ML

### Milestone reached
- All 71 first-round `(U,V)` parameter pairs have now been completed.
- The workflow to generate `rho_G` and `re_G` from raw Lanczos data is ready.
- The project is now prepared to begin machine learning directly on `rho_G` and `re_G`.

### Data representation for learning
At this stage, the natural ML input is no longer the raw continued-fraction output itself, but the processed spectral representation:
- `rho_G`: spectral-weight information derived from the Green function
- `re_G`: real-part information derived from the Green function

This is an important transition point because the learning problem is now defined on a standardized, physics-informed representation rather than on raw Lanczos intermediates.

### Immediate ML strategy
The first ML pass should stay simple and interpretable:

1. Build a single consistent dataset over all 71 `(U,V)` points using matched `rho_G` and `re_G` grids.
2. Standardize metadata and tensor layout so every sample has the same ordering in frequency and momentum-path coordinates.
3. Run baseline exploratory learning:
   - PCA / UMAP on `rho_G` only
   - PCA / UMAP on `re_G` only
   - PCA / UMAP on the concatenated `rho_G + re_G` representation
4. Train simple regressors to predict `U`, `V`, or both from the processed spectra.
5. Compare which representation carries more usable physics information:
   - `rho_G` alone
   - `re_G` alone
   - combined `rho_G` and `re_G`

### Key questions for this phase
- Is `rho_G` alone already sufficient to organize the 71 parameter points meaningfully?
- Does adding `re_G` improve reconstruction of `(U,V)` or sharpen clustering structure?
- Do the learned embeddings align smoothly with interaction strength or suggest regime boundaries?
- Are there spectral families that appear robust across nearby `(U,V)` values?

### Practical caution
Because the dataset still contains only 71 parameter points, the first ML round should prioritize:
- low-capacity models
- dimensionality reduction
- careful train/validation splits
- physical interpretability over raw predictive performance

### Next concrete work item
Construct the first ML-ready dataset object that packages:
- `(U,V)` labels
- `rho_G`
- `re_G`
- path/frequency axes
- preprocessing metadata

Once that dataset object is stable, baseline learning experiments can begin immediately.

## 2026-04-26 | First baseline ML pass on rho_G/re_G

### Dataset package
- Source file: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- ML-ready output: `data/Gcalpath/rho_re_g/ml_baseline_v2/ml_ready_dataset.npz`
- Samples: 71 `(U,V)` parameter points
- Tensor layout: `(sample, k_path, omega)`
- Shapes: `rho_G = (71, 121, 301)`, `reG = (71, 121, 301)`
- Axes: 121-point special momentum path, 301-point omega grid from `-15` to `15`

### Baseline method
- Representations tested: `rho_G`, `reG`, and concatenated `rho_G + reG`
- Dimensionality reduction: PCA with 8 components
- Regression baseline: ridge regression on train-only PCA scores
- Split: deterministic holdout with 56 train and 15 test samples
- Output folder: `data/Gcalpath/rho_re_g/ml_baseline_v2`

### First numerical readout
- `rho_G`: first 8 PCs explain 76.0% of total variance; ridge test `R2_V = 0.507`, `R2_U = -0.491`
- `reG`: first 8 PCs explain 81.3% of total variance; ridge test `R2_V = 0.259`, `R2_U = -0.186`
- `rho_G + reG`: first 8 PCs explain 77.3% of total variance; ridge test `R2_V = 0.509`, `R2_U = -0.306`

### Interpretation
The first-pass PCA representation contains clear low-dimensional spectral structure, but the simple PCA-plus-ridge model is much better at reconstructing `V` than `U` on this particular holdout split. This suggests the first follow-up should inspect PCA embeddings and spectral-derived physics features before increasing model capacity.

### Next action
Add physics-informed features such as gap estimates, low-frequency spectral weight, spectral moments, and peak locations, then compare regressors using repeated or leave-one-out validation.

## 2026-04-27 | Physics-feature baseline added

### Feature extraction
- Script: `create_code/ml_rho_re_g_physics_features.py`
- Input: `data/Gcalpath/rho_re_g/ml_baseline_v2/ml_ready_dataset.npz`
- Output folder: `data/Gcalpath/rho_re_g/ml_physics_features`
- Feature table: `physics_features.csv`
- Feature tensor: `physics_features.npz`
- Total features: 93
- `rho_G` features: 55
- `reG` features: 38

### Feature families
- Low-energy spectral weight around omega = 0
- Estimated path-averaged gap thresholds
- Positive/negative integrated spectral weight
- Particle-hole asymmetry
- Spectral centroids and widths
- Peak positions and peak values along the k path
- `reG` zero-frequency values, slopes, norms, sign changes, and extrema

### Model comparison
The same deterministic 56/15 holdout split was used for direct comparison with the PCA baseline, plus shuffled 5-fold cross-validation.

- `rho_G` features + ridge:
  - Holdout: `R2_U = 0.839`, `R2_V = 0.990`
  - 5-fold CV: `R2_U = 0.923`, `R2_V = 0.991`
- `reG` features + ridge:
  - Holdout: `R2_U = 0.392`, `R2_V = 0.978`
  - 5-fold CV: `R2_U = 0.896`, `R2_V = 0.982`
- Combined features + ridge:
  - Holdout: `R2_U = 0.774`, `R2_V = 0.992`
  - 5-fold CV: `R2_U = 0.915`, `R2_V = 0.991`

Random forests were also tested, but ridge regression performed better on this compact, interpretable feature set.

### Interpretation
The failure of PCA-score ridge regression to estimate `U` was not because `U` is absent from the spectra. Physics-inspired scalar features recover both `U` and `V` much more cleanly. The strongest ridge features suggest that `U` is tied to gap/peak/intensity redistribution, while `V` is strongly tied to spectral centroid shifts and particle-hole asymmetry.

### Next action
Inspect the feature maps over the `(U,V)` plane and use the top-ranked features to define candidate crossover diagnostics before trying higher-capacity neural models.

## 2026-04-27 | Scope narrowed to positive V

### Direction change
Future data generation will focus on positive nearest-neighbor interaction `V`. The negative-`V` data from round 1 remains useful as a contrast set, but the next calculations should prioritize the repulsive-`V` side where charge redistribution and extended-Hubbard competition are the central physics questions.

### Proposed next positive-V sweep
- Candidate list: `create_code/UVset_positive_next.txt`
- Tier 1 fills the missing `U=11` positive-`V` plane.
- Tier 2 densifies the region near `V ~ U/4`.
- Tier 3 adds half-integer `U` interpolation probes with positive `V`.

### ML implication
Validation should shift from random holdout only to positive-`V` generalization tests:
- train on integer `U`, test on half-integer `U`
- train with sparse positive `V`, test on dense `V ~ U/4` points
- compare feature trends along fixed-`U` positive-`V` cuts

## 2026-04-30 | Expanded dataset rerun: full and positive-V learning

### Expanded dataset
- Source: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- Regenerated tensor package: `data/Gcalpath/rho_re_g/ml_baseline_expanded/ml_ready_dataset.npz`
- Total samples: 120 `(U,V)` points
- Positive-only samples: 49 points with `V > 0`

### Full-dataset learning
- Output folder: `data/Gcalpath/rho_re_g/ml_physics_features_expanded_full`
- Best compact model: `rho_G` features + ridge
  - Holdout: `R2_U = 0.984`, `R2_V = 0.997`
  - 5-fold CV: `R2_U = 0.949`, `R2_V = 0.996`
- Combined features + ridge is also strong:
  - Holdout: `R2_U = 0.970`, `R2_V = 0.996`
  - 5-fold CV: `R2_U = 0.953`, `R2_V = 0.996`

### Positive-V-only learning
- Output folder: `data/Gcalpath/rho_re_g/ml_physics_features_expanded_positive`
- Filter: `V >= 0.001`, excluding the `V = 0` line
- Best stable model: `rho_G` features + ridge
  - Holdout: `R2_U = 0.824`, `R2_V = 0.999`
  - 5-fold CV: `R2_U = 0.918`, `R2_V = 0.996`
- Combined features + ridge has excellent holdout performance but unstable 5-fold `U` behavior:
  - Holdout: `R2_U = 0.854`, `R2_V = 0.998`
  - 5-fold CV: `R2_U = -0.095`, `R2_V = 0.987`

### Interpretation
The full expanded dataset now gives very strong parameter recovery from interpretable features. Positive-`V` learning remains excellent for `V`, but `U` is less stable because the positive-only grid is sparser and uneven: several fixed-`U` cuts have only one or two positive-`V` points. The largest positive-only holdout `U` errors occur near `(10,3)`, `(8.5,2.5)`, `(4,2)`, and the low-`U` positive boundary.

### Recommended next positive-V data
- Candidate list: `create_code/UVset_positive_recommended_after_expansion.txt`
- Tier 1 targets local neighborhoods around the largest positive-only learning errors.
- Tier 2 fills sparse fixed-`U` positive-`V` cuts if Tier 1 is not enough.

## 2026-04-30 | Rerun after omega-grid expansion

### Updated omega grid
- Source: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- Regenerated tensor package: `data/Gcalpath/rho_re_g/ml_baseline_expanded_w1201/ml_ready_dataset.npz`
- Total samples: 120 `(U,V)` points
- Positive-only samples: 49 points with `V > 0`
- Omega points: 1201, spanning `[-15, 15]`

### Full-dataset learning
- Output folder: `data/Gcalpath/rho_re_g/ml_physics_features_expanded_full_w1201`
- Best compact model: `rho_G` features + ridge
  - Holdout: `R2_U = 0.988`, `R2_V = 0.997`
  - 5-fold CV: `R2_U = 0.958`, `R2_V = 0.992`
- Combined features + ridge:
  - Holdout: `R2_U = 0.984`, `R2_V = 0.996`
  - 5-fold CV: `R2_U = 0.900`, `R2_V = 0.995`

### Positive-V-only learning
- Output folder: `data/Gcalpath/rho_re_g/ml_physics_features_expanded_positive_w1201`
- Filter: `V >= 0.001`, excluding the `V = 0` line
- Best stable model: `rho_G` features + ridge
  - Holdout: `R2_U = 0.841`, `R2_V = 0.998`
  - 5-fold CV: `R2_U = 0.922`, `R2_V = 0.997`
- Combined features + ridge is now more stable than before:
  - Holdout: `R2_U = 0.848`, `R2_V = 0.998`
  - 5-fold CV: `R2_U = 0.833`, `R2_V = 0.995`

### Interpretation
The denser omega grid improves the robustness of the interpretable feature learning, especially for full-data `U` prediction and the positive-only combined-feature cross-validation. The positive-only case remains very strong for `V`, while `U` is still limited by sparse positive-`V` coverage rather than frequency resolution alone.

## 2026-04-30 | Luttinger-path learning from reG(k,0)

### Analysis target
- Script: `create_code/ml_luttinger_path_reG0.py`
- Dataset: `data/Gcalpath/rho_re_g/ml_baseline_expanded_w1201/ml_ready_dataset.npz`
- Definition used: `reG(k, omega=0) = 0`, equivalent to `Re G(k,0) = 0` up to the stored `-2` convention
- Scope: crossings along the 121-point `Gamma-X-M-Gamma` path, not a full 2D Brillouin-zone surface

### Outputs
- Full-data folder: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_full_w1201`
- Positive-`V` folder: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_positive_w1201`
- Main table in each folder: `luttinger_crossings.csv`
- Feature package in each folder: `luttinger_path_features.npz`

### Numerical result
Every current sample has two `reG(k,0)=0` crossings along the path:
- one crossing pinned at the `X` point/path boundary near path coordinate `41`
- one crossing on the `M-Gamma` leg near path coordinate `109.284`

Because the crossing count and positions are nearly fixed, the Luttinger-path features are weaker predictors than the full spectral features.

- Full data, ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.881`, `R2_V = 0.014`
  - 5-fold CV: `R2_U = 0.669`, `R2_V = 0.706`
- Positive `V`, ridge on Luttinger-path features:
  - Holdout: `R2_U = -4.693`, `R2_V = 0.779`
  - 5-fold CV: `R2_U = -0.296`, `R2_V = 0.844`

### Interpretation
Along this high-symmetry path, the Luttinger zero locations themselves do not move enough to organize the positive-`V` physics. The more useful information is the spectral weight and line shape near these pinned crossings. A full 2D Luttinger-surface study would require `reG(kx,ky,0)` over a 2D momentum grid.

## 2026-05-04 | Current dataset rerun after new positive-V additions

### Current dataset
- Source: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- Regenerated tensor package: `data/Gcalpath/rho_re_g/ml_baseline_current_20260504/ml_ready_dataset.npz`
- Total samples: 160 `(U,V)` points
- Positive-only samples: 89 points with `V > 0`
- Omega points: 1201, spanning `[-15, 15]`

### Physics-feature learning
- Full-data output: `data/Gcalpath/rho_re_g/ml_physics_features_current_full_20260504`
- Positive-`V` output: `data/Gcalpath/rho_re_g/ml_physics_features_current_positive_20260504`
- Full data, combined features + ridge:
  - Holdout: `R2_U = 0.983`, `R2_V = 0.999`
  - 5-fold CV: `R2_U = 0.976`, `R2_V = 0.997`
- Full data, `rho_G` features + ridge:
  - Holdout: `R2_U = 0.966`, `R2_V = 0.998`
  - 5-fold CV: `R2_U = 0.962`, `R2_V = 0.996`
- Positive `V`, `rho_G` features + ridge:
  - Holdout: `R2_U = 0.948`, `R2_V = 0.987`
  - 5-fold CV: `R2_U = 0.947`, `R2_V = 0.995`
- Positive `V`, combined features + ridge:
  - Holdout: `R2_U = 0.932`, `R2_V = 0.992`
  - 5-fold CV: `R2_U = 0.928`, `R2_V = 0.982`

### Luttinger-path reG(k,0) learning
- Full-data output: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_current_full_20260504`
- Positive-`V` output: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_current_positive_20260504`
- Every current sample still has two `reG(k,0)=0` path crossings.
- Full data, ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.568`, `R2_V = 0.731`
  - 5-fold CV: `R2_U = 0.606`, `R2_V = 0.695`
- Positive `V`, ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.655`, `R2_V = 0.738`
  - 5-fold CV: `R2_U = 0.525`, `R2_V = 0.852`

### Interpretation
The added positive-`V` data substantially improves positive-only physics-feature learning, especially `U` stability. The current best positive-only model is now `rho_G` features + ridge, with cross-validated `R2_U = 0.947` and `R2_V = 0.995`. Luttinger-path features are more useful than before for positive `V`, but remain secondary because the two crossing locations are still nearly pinned along the path.

## 2026-05-04 | eta=0.1 rerun and gap analysis

### eta=0.1 dataset
- Source: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- Regenerated tensor package: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260504/ml_ready_dataset.npz`
- Total samples: 160 `(U,V)` points
- Positive-only samples: 89 points with `V > 0`
- Omega points: 1201, spanning `[-15, 15]`
- Broadening: `eta = 0.1`

### Physics-feature learning
- Full-data output: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_full_20260504`
- Positive-`V` output: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_positive_20260504`
- Full data, `rho_G` features + ridge:
  - Holdout: `R2_U = 0.938`, `R2_V = 0.997`
  - 5-fold CV: `R2_U = 0.942`, `R2_V = 0.996`
- Full data, combined features + ridge:
  - Holdout: `R2_U = 0.952`, `R2_V = 0.998`
  - 5-fold CV: `R2_U = 0.933`, `R2_V = 0.997`
- Positive `V`, `rho_G` features + ridge:
  - Holdout: `R2_U = 0.891`, `R2_V = 0.993`
  - 5-fold CV: `R2_U = 0.897`, `R2_V = 0.994`

### Luttinger-path reG(k,0)
- Full-data output: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_full_20260504`
- Positive-`V` output: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_positive_20260504`
- Every sample still has two path crossings.
- Full data, 5-fold CV: `R2_U = 0.619`, `R2_V = 0.350`
- Positive `V`, 5-fold CV: `R2_U = 0.323`, `R2_V = 0.763`

### Positive-V gap analysis
- Output folder: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_20260504`
- Metric: `min_{omega in [-1,1]} <rho_G>_path / max_omega <rho_G>_path`
- Positive-`V` samples: 89
- Ratio statistics:
  - min: `0.000378`
  - median: `0.00452`
  - max: `0.505`
- Threshold counts:
  - ratio `< 0.001`: 19 / 89
  - ratio `< 0.01`: 64 / 89
  - ratio `< 0.05`: 78 / 89
  - ratio `< 0.10`: 82 / 89
  - ratio `< 0.20`: 84 / 89

### Interpretation
Reducing eta from `0.2` to `0.1` sharpens the gap diagnosis substantially. The positive-`V` spectra now show much stronger low-energy suppression overall, but the lowest `U` and lowest positive `V` points remain not cleanly gapped. The eta=0.1 feature learning remains strong, though slightly less stable for positive-only `U` than the eta=0.2 current run, likely because sharper spectra make interpolation more sensitive.

## 2026-05-04 | Adopted max-k gap proxy and filtered Luttinger learning

### Gap-opening definition
Use the conservative positive-`V` gap proxy:

```text
max_k rho_G(k, omega=0) / max_omega <rho_G>_path
```

Working definitions:
- gap opening: ratio `< 0.10`
- Luttinger-path learning subset: ratio `< 0.01`

### Classification result at eta=0.1
- Output folder: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_20260504`
- Classification table: `positive_v_gap_classification_maxk_rho0.csv`
- Positive-`V` samples: 89
- Gap-open samples, ratio `< 0.10`: 75 / 89
- Luttinger-learning samples, ratio `< 0.01`: 55 / 89
- Not gap-open samples, ratio `>= 0.10`: 14 / 89

### Luttinger learning on gapped positive-V subset
- Output folder: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_positive_gaplt0p01_20260504`
- Filter: `V > 0` and max-k gap proxy `< 0.01`
- Samples: 55
- Every sample still has two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.763`, `R2_V = 0.893`
  - 5-fold CV: `R2_U = 0.699`, `R2_V = 0.949`

### Interpretation
Restricting Luttinger-path learning to strongly gap-open positive-`V` spectra makes the Luttinger signal cleaner, especially for `V`. The zero-crossing count is still fixed, so the learning comes from crossing-associated line-shape information and small variations in the path features rather than from the appearance/disappearance of crossings.

## 2026-05-05 | Unsupervised eta=0.1 regime learning

### Workflow
- Added script: `create_code/ml_unsupervised_eta0p1.py`
- Output folder: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_20260505`
- Inputs:
  - `ml_physics_features_eta0p1_full_20260504/physics_features.npz`
  - `ml_physics_features_eta0p1_positive_20260504/physics_features.npz`
  - `gap_analysis_eta0p1_20260504/positive_v_gap_classification_maxk_rho0.csv`
  - `ml_luttinger_path_reG0_eta0p1_positive_gaplt0p01_20260504/luttinger_path_features.npz`
- Clustering used standardized rho/ReG physics features only. `U`, `V`, gap ratio, and Luttinger features were used only after fitting for interpretation.

### Positive-`V` result
- Samples: 89
- Combined-feature PCA variance fractions: `0.598`, `0.195`, `0.080`
- Best KMeans by silhouette: `k=2`
- Metrics:
  - silhouette: `0.523`
  - Calinski-Harabasz: `60.8`
  - Davies-Bouldin: `0.812`
  - seed-stability adjusted Rand index: `0.924`
- Cluster 0:
  - 78 samples
  - `U` median `8.5`, `V` median `2.344`
  - median max-k gap ratio `0.00327`
  - gap-open fraction `0.962`
  - strong-gap fraction `0.705`
- Cluster 1:
  - 11 samples
  - `U` median `2.5`, `V` median `0.625`
  - median max-k gap ratio `0.964`
  - gap-open fraction `0.0`
  - strong-gap fraction `0.0`

### Subset checks
- Full-data combined features also prefer `k=2`, silhouette `0.450`, stability ARI `1.000`.
- Positive gap-open subset prefers `k=2`, silhouette `0.426`, stability ARI `1.000`.
- Positive strong-gap subset prefers `k=4`, silhouette `0.477`, stability ARI `0.964`.

### Interpretation
Unsupervised learning recovers the main physics boundary without being given `U`, `V`, or the gap label: positive-`V` data split into a small low-`U`/low-`V` non-gapped group and a large mostly gapped group. Within the already strong-gap subset, the best clustering becomes fourfold, suggesting additional internal structure in the gapped phase rather than a single uniform regime.

### Improvement pass
- Added raw low-energy spectrum PCA representations using `|omega| <= 1`:
  - `raw_rho_low_w1_pca`
  - `raw_reG_low_w1_pca`
  - `raw_combined_low_w1_pca`
- Added consensus diagnostics across best KMeans, GMM, and agglomerative labels.
- Added `luttinger_geometry` clustering only for the positive-`V` strong-gap subset.
- Added plots for raw-combined and Luttinger-geometry cluster maps.

### Improved unsupervised findings
- Positive `V`, raw combined low-energy PCA:
  - best KMeans: `k=3`
  - silhouette: `0.633`
  - stability ARI: `0.790`
  - This suggests the raw low-energy spectra split the positive-`V` data into finer structure than the interpretable physics features alone.
- Positive `V`, raw rho low-energy PCA:
  - best KMeans: `k=5`
  - silhouette: `0.701`
  - stability ARI: `0.985`
  - This is the cleanest mathematical clustering, but it should be interpreted carefully because it may emphasize spectral-shape details more than coarse phase regimes.
- Positive gap-open subset, raw combined low-energy PCA:
  - best KMeans: `k=3`
  - silhouette: `0.539`
  - stability ARI: `1.000`
- Positive strong-gap subset:
  - combined physics features prefer `k=4`, silhouette `0.477`
  - raw combined low-energy PCA prefers `k=2`, silhouette `0.575`, stability ARI `1.000`
  - Luttinger geometry prefers `k=2`, silhouette `0.507`, stability ARI `1.000`, consensus pairwise ARI `0.952`

### Updated interpretation
The robust coarse physics remains two-stage: a non-gapped low-`U`/low-`V` region separates from the gapped region, and the strong-gap region has additional internal structure. Raw low-energy rho spectra give the sharpest clusters, while Luttinger geometry gives a very stable two-way split inside the strongly gapped subset. The next physics question is whether the raw-rho `k=5` split corresponds to meaningful sub-regimes or mainly to smooth spectral evolution across `(U,V)`.

## 2026-05-05 | Gapped-region focused unsupervised refinement

### Workflow updates
- Extended `create_code/ml_unsupervised_eta0p1.py` for gapped-region learning.
- Added multi-window raw-spectrum PCA representations:
  - `|omega| <= 0.25`
  - `|omega| <= 0.5`
  - `|omega| <= 1.0`
  - `|omega| <= 2.0`
- Added `raw_combined_multiscale_low_pca`, concatenating the raw combined PCA coordinates across all four windows.
- Added a deep-gap subset:
  - `positive_v_deep_gap`: max-k gap ratio `< 0.001`
  - sample count: 17
- Added `gapped_recommendations.csv`, ranking gapped-region KMeans results by silhouette, seed stability, and cross-method consensus.

### Best gapped-region results
- Gap-open positive-`V` subset, ratio `< 0.10`, 75 samples:
  - best recommendation: `raw_rho_low_w2p0_pca`
  - KMeans: `k=2`
  - silhouette: `0.722`
  - stability ARI: `1.000`
  - consensus pairwise ARI: `0.785`
- Strong-gap positive-`V` subset, ratio `< 0.01`, 55 samples:
  - best recommendation: `raw_reG_low_w0p5_pca`
  - KMeans: `k=2`
  - silhouette: `0.648`
  - stability ARI: `1.000`
  - consensus pairwise ARI: `0.732`
- Deep-gap positive-`V` subset, ratio `< 0.001`, 17 samples:
  - best recommendation: `raw_rho_low_w2p0_pca`
  - KMeans: `k=2`
  - silhouette: `0.601`
  - stability ARI: `1.000`
  - consensus pairwise ARI: `0.843`

### Gapped-region interpretation
The most robust gapped-region structure is a two-way split, not a many-cluster split. Across gap-open, strong-gap, and deep-gap subsets, the stable division is roughly between a moderate-gap/lower-`V` sector and a deeper-gap/higher-`V` sector. In the strong-gap subset, the best signal comes from low-energy `ReG` within `|omega| <= 0.5`, consistent with the idea that the internal structure of the gapped phase is tied to low-energy real-part/Luttinger geometry rather than only total spectral-weight suppression.

## 2026-05-05 | Reanalysis including `V = 0`

### Dataset and gap analysis
- New working scope: `V >= 0`
- Samples:
  - total dataset: 160
  - `V > 0`: 89
  - `V = 0`: 22
  - `V >= 0`: 111
- Gap output: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260505`
- Gap proxy: `max_k rho_G(k,0) / max_omega <rho_G>_path`
- Counts:
  - gap-open, ratio `< 0.10`: 87 / 111
  - strong-gap, ratio `< 0.01`: 58 / 111
  - deep-gap, ratio `< 0.001`: 17 / 111
  - not gap-open: 24 / 111

### Supervised learning
- Output: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260505`
- Samples: 111
- `rho_G` ridge, five-fold CV:
  - `R2_U = 0.922`
  - `R2_V = 0.995`
- `rho_G` random forest, five-fold CV:
  - `R2_U = 0.936`
  - `R2_V = 0.904`
- Combined random forest, five-fold CV:
  - `R2_U = 0.948`
  - `R2_V = 0.910`
- Ridge models using only `reG` or combined features became numerically unstable after including the `V=0` line, so the manuscript now emphasizes `rho_G` ridge and random-forest baselines.

### Luttinger-path learning
- Output: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260505`
- Filter: `V >= 0` and max-k gap proxy `< 0.01`
- Samples: 58
- Every sample has two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.781`, `R2_V = 0.941`
  - 5-fold CV: `R2_U = 0.606`, `R2_V = 0.899`

### Unsupervised learning
- Output: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260505`
- Coarse `V >= 0` combined-feature KMeans:
  - best `k=2`
  - silhouette `0.467`
  - stability ARI `0.969`
  - cluster 0: 91 samples, mostly gapped, median gap ratio `0.00610`
  - cluster 1: 20 samples, non-gapped, median gap ratio `1.078`
- Gapped-region recommendations:
  - gap-open subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.685`, stability ARI `1.000`
  - strong-gap subset: raw `reG`, `|omega| <= 1.0`, `k=2`, silhouette `0.637`, stability ARI `1.000`
  - deep-gap subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.601`, stability ARI `1.000`

### Manuscript update
- Updated LaTeX manuscript to the nonnegative-`V` analysis:
  - `manuscript_positive_v_ml/main.tex`
  - `manuscript_positive_v_ml/main.pdf`
- Main interpretation is unchanged but softened: including `V=0` makes the coarse non-gapped/gapped boundary more continuous, while the robust two-way internal structure of the gapped region survives.

## 2026-05-07 | Rerun of supervised and unsupervised `V >= 0` analysis

### Scope
- Requested rerun using the current eta=`0.1` canonical dataset.
- Manuscript scope remains strictly `V >= 0`; no below-zero-`V` data are used in manuscript figures.
- Dataset: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260504/ml_ready_dataset.npz`
- Nonnegative-`V` samples: 111

### Fresh outputs
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260507`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260507`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260507`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260507`

### Rerun results
- Gap counts match the previous nonnegative-`V` analysis:
  - gap-open, ratio `< 0.10`: 87 / 111
  - strong-gap, ratio `< 0.01`: 58 / 111
  - deep-gap, ratio `< 0.001`: 17 / 111
  - not gap-open: 24 / 111
- Supervised five-fold CV:
  - `rho_G` ridge: `R2_U = 0.922`, `R2_V = 0.995`
  - `rho_G` random forest: `R2_U = 0.936`, `R2_V = 0.904`
  - `reG` random forest: `R2_U = 0.932`, `R2_V = 0.891`
  - combined random forest: `R2_U = 0.948`, `R2_V = 0.910`
- Luttinger-path learning in the strong-gap subset:
  - 58 samples, all with two zero crossings
  - five-fold CV: `R2_U = 0.606`, `R2_V = 0.899`
- Unsupervised learning:
  - full nonnegative-`V` combined-feature KMeans: `k=2`, silhouette `0.467`, stability ARI `0.969`
  - gap-open best recommendation: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.685`
  - strong-gap best recommendation: raw `reG`, `|omega| <= 1.0`, `k=2`, silhouette `0.637`
  - deep-gap best recommendation: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.601`

### Manuscript update
- Updated `manuscript_positive_v_ml/main.tex` to point at the 20260507 figure outputs.
- Replaced the stale Markdown draft with a pointer to the LaTeX manuscript.

## 2026-05-07 | Kernel/SVM unsupervised refinement for `V >= 0`

### Workflow
- Added script: `create_code/ml_unsupervised_kernel_eta0p1.py`
- Output: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260507`
- Scope:
  - only `V >= 0`
  - focus subsets: gap-open (`R0 < 0.10`) and strong-gap (`R0 < 0.01`)
  - `U`, `V`, gap labels, and Luttinger labels are used only for interpretation/recommendations, not clustering inputs
- Methods added:
  - RBF kernel PCA + KMeans
  - RBF spectral clustering
  - RBF One-Class SVM boundary/anomaly scoring
  - bootstrap stability using 40 subsamples at 80% sample fraction

### Main refinement result
- Strong-gap subset, raw `reG`, `|omega| <= 1.0`, spectral clustering:
  - `k = 2`
  - silhouette `0.643`
  - bootstrap ARI `0.959`
  - cluster 0: 33 samples, median `U = 10.0`, median `V = 2.0`, median `R0 = 0.00519`, deep-gap fraction `0.0`
  - cluster 1: 25 samples, median `U = 9.0`, median `V = 3.5`, median `R0 = 0.000742`, deep-gap fraction `0.68`
- Luttinger-geometry spectral clustering in the same strong-gap subset:
  - `k = 2`
  - silhouette `0.516`
  - bootstrap ARI `0.974`

### Interpretation
The kernel/SVM refinement confirms that the strongest nonlinear unsupervised signal is still a two-way split inside the strong-gap regime. Low-energy `reG` remains the best representation. Luttinger geometry gives a less separated but very stable two-way split, supporting the interpretation that the internal gapped-regime structure is tied to zero-surface geometry.

### Additional data recommendations
- Recommendation CSV: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260507/additional_data_recommendations.csv`
- Highest-priority points target:
  - the strong-gap spectral-cluster boundary
  - the deep-gap threshold near `R0 = 0.001`
  - the strong-gap threshold near `R0 = 0.01`
  - One-Class-SVM boundary/outlier neighborhoods

## 2026-05-11 | Updated-data rerun of supervised and unsupervised `V >= 0` analysis

### Updated dataset
- Source combined data: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- New ML-ready dataset: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260511/ml_ready_dataset.npz`
- Total sample count in combined file: 196
- `V >= 0` sample count: 147
  - `V > 0`: 123
  - `V = 0`: 24
- All 36 previously recommended extra `(U,V)` points are present in the updated data.

### Fresh outputs
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260511`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260511`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260511`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260511`
- Kernel/SVM refinement: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260511`
- New suggested UV set: `create_code/UVset_nonnegative_kernel_unsupervised_20260511.txt`

### Gap counts
- Gap-open, ratio `< 0.10`: 122 / 147
- Strong-gap, ratio `< 0.01`: 87 / 147
- Deep-gap, ratio `< 0.001`: 17 / 147
- Not gap-open: 25 / 147
- Median ratio: `0.006818`

### Supervised learning, five-fold CV
- `rho_G` ridge: `R2_U = 0.395`, `R2_V = 0.997`
  - Interpretation: linear `rho_G` features still learn `V` extremely well, but the new boundary points make the `U` direction nonlinear.
- `rho_G` random forest: `R2_U = 0.942`, `R2_V = 0.919`
- `reG` random forest: `R2_U = 0.944`, `R2_V = 0.895`
- combined random forest: `R2_U = 0.956`, `R2_V = 0.919`
  - This is the best robust supervised baseline after the data expansion.

### Luttinger-path learning
- Strong-gap subset: 87 samples, all with two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.918`, `R2_V = 0.986`
  - 5-fold CV: `R2_U = 0.750`, `R2_V = 0.943`
- The extra data substantially improve the learnability of the Luttinger-surface geometry.

### Unsupervised learning
- Full `V >= 0` combined-feature KMeans:
  - `k=2`, silhouette `0.503`, stability ARI `0.957`
  - cluster 0: 126 samples, mostly gapped, median `R0 = 0.00519`
  - cluster 1: 21 samples, low-`U`/low-`V`, median `R0 = 0.964`
- Gapped-region KMeans recommendations:
  - gap-open subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.706`, stability ARI `1.000`
  - strong-gap subset: raw `reG`, `|omega| <= 0.5`, `k=2`, silhouette `0.611`, stability ARI `1.000`, consensus ARI `0.969`
  - deep-gap subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.601`, stability ARI `1.000`
- Strong-gap two-cluster summary from raw `reG`, `|omega| <= 0.5`:
  - moderate-gap/lower-`V`: 40 samples, median `U=9.5`, median `V=2.0`, median `R0=0.00592`, deep fraction `0.0`
  - deeper-gap/higher-`V`: 47 samples, median `U=8.5`, median `V=3.25`, median `R0=0.00116`, deep fraction `0.362`

### Kernel/SVM refinement
- Best overall kernel metric:
  - gap-open subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=4`
  - silhouette `0.693`, bootstrap ARI `0.913`
- Full `V >= 0`, raw `reG`, `|omega| <= 1.0`, kernel PCA + KMeans, `k=2`:
  - silhouette `0.588`, bootstrap ARI `0.996`
- Strong-gap subset, raw `reG`, `|omega| <= 1.0`, kernel PCA + KMeans, `k=4`:
  - silhouette `0.526`, bootstrap ARI `0.993`
- Strong-gap Luttinger geometry, kernel PCA + KMeans, `k=2`:
  - silhouette `0.508`, bootstrap ARI `1.000`

### Additional data recommendations
- 36 new suggestions were written to `create_code/UVset_nonnegative_kernel_unsupervised_20260511.txt`.
- The top candidates are:
  - `(5.375,2.328)`, `(7.875,2.578)`, `(10.750,3.000)`, `(4.000,1.438)`, `(10.000,2.875)`, `(4.000,1.188)`, `(12.000,3.125)`, `(6.250,2.516)`, `(4.500,2.125)`, `(9.500,2.875)`, `(2.875,1.250)`, `(5.000,2.266)`
- These target the strong-gap internal boundary, the `R0 ~ 0.01` threshold, the gap-opening threshold, and one-class-SVM boundary neighborhoods.

### Manuscript update
- Updated `manuscript_positive_v_ml/main.tex` to use the 20260511 outputs and to keep the manuscript scope strictly `V >= 0`.

## 2026-05-15 | Rerun after adding the 20260511 recommended `V >= 0` data

### Updated dataset
- Source combined data: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- New ML-ready dataset: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260515/ml_ready_dataset.npz`
- Total sample count in combined file: 232
- `V >= 0` sample count: 183
  - `V > 0`: 157
  - `V = 0`: 26
- All 36 points from `create_code/UVset_nonnegative_kernel_unsupervised_20260511.txt` are present.

### Fresh outputs
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260515`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260515`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260515`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260515`
- Kernel/SVM refinement: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260515`
- New suggested UV set: `create_code/UVset_nonnegative_kernel_unsupervised_20260515.txt`

### Gap counts
- Gap-open, ratio `< 0.10`: 157 / 183
- Strong-gap, ratio `< 0.01`: 110 / 183
- Deep-gap, ratio `< 0.001`: 17 / 183
- Not gap-open: 26 / 183
- Median ratio: `0.006336`

### Supervised learning, five-fold CV
- `rho_G` ridge: `R2_U = 0.443`, `R2_V = 0.996`
- `rho_G` random forest: `R2_U = 0.955`, `R2_V = 0.929`
- `reG` random forest: `R2_U = 0.947`, `R2_V = 0.886`
- combined random forest: `R2_U = 0.961`, `R2_V = 0.911`
- The best robust supervised baseline remains the combined random forest; `rho_G` random forest gives the best `V` among the random-forest models.

### Luttinger-path learning
- Strong-gap subset: 110 samples, all with two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.821`, `R2_V = 0.971`
  - 5-fold CV: `R2_U = 0.822`, `R2_V = 0.947`

### Unsupervised learning
- Full `V >= 0` combined-feature KMeans:
  - `k=2`, silhouette `0.559`, stability ARI `0.995`
  - mostly gapped cluster: 165 samples, median `R0 = 0.00519`
  - low-`U`/low-`V` cluster: 18 samples, median `R0 = 1.332`
- Gapped-region recommendations:
  - gap-open subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.720`, stability ARI `1.000`
  - strong-gap subset: raw combined, `|omega| <= 2.0`, `k=2`, silhouette `0.547`, stability ARI `1.000`, consensus ARI `0.881`
  - strong-gap raw `reG`, `|omega| <= 2.0`, `k=2`, silhouette `0.556`, stability ARI `1.000`, consensus ARI `0.815`
  - deep-gap subset remains 17 samples and still prefers raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.601`

### Kernel/SVM refinement
- Best overall kernel metric:
  - gap-open subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=4`, gamma factor `0.5`
  - silhouette `0.710`, bootstrap ARI `0.930`
- Full `V >= 0`, raw `reG`, `|omega| <= 1.0`, kernel PCA + KMeans, `k=2`:
  - silhouette `0.629`, bootstrap ARI `0.967`
- Strong-gap subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=2`:
  - silhouette `0.606`, bootstrap ARI `0.976`

### Additional data recommendations
- 36 new suggestions were written to `create_code/UVset_nonnegative_kernel_unsupervised_20260515.txt`.
- Top candidates:
  - `(7.938,2.539)`, `(4.000,1.562)`, `(4.500,2.062)`, `(10.000,2.812)`, `(10.500,2.938)`, `(12.000,3.062)`, `(3.125,1.453)`, `(4.938,2.062)`, `(4.000,1.469)`, `(4.000,1.156)`
- The new recommendations are now finer local refinements rather than broad missing-region requests, indicating that the previous additional-data round successfully filled the main boundaries.

## 2026-05-17 | Rerun after adding the 20260515 recommended `V >= 0` data

### Updated dataset
- Source combined data: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- New ML-ready dataset: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260517/ml_ready_dataset.npz`
- Total sample count in combined file: 268
- `V >= 0` sample count: 219
  - `V > 0`: 191
  - `V = 0`: 28
- All 36 points from `create_code/UVset_additional_nonnegative_20260515.txt` are present.

### Fresh outputs
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260517`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260517`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260517`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260517`
- Kernel/SVM refinement: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260517`
- New suggested UV set: `create_code/UVset_additional_nonnegative_20260517.txt`

### Gap counts
- Gap-open, ratio `< 0.10`: 191 / 219
- Strong-gap, ratio `< 0.01`: 136 / 219
- Deep-gap, ratio `< 0.001`: 22 / 219
- Not gap-open: 28 / 219
- Median ratio: `0.006336`

### Supervised learning, five-fold CV
- `rho_G` ridge: `R2_U = 0.907`, `R2_V = 0.997`
- `rho_G` random forest: `R2_U = 0.958`, `R2_V = 0.941`
- `reG` random forest: `R2_U = 0.947`, `R2_V = 0.903`
- combined random forest: `R2_U = 0.958`, `R2_V = 0.919`
- The new data strongly improve the linear `rho_G` ridge `U` learning compared with the 20260515 run.

### Luttinger-path learning
- Strong-gap subset: 136 samples, all with two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.831`, `R2_V = 0.967`
  - 5-fold CV: `R2_U = 0.860`, `R2_V = 0.933`

### Unsupervised learning
- Full `V >= 0` combined-feature KMeans:
  - `k=2`, silhouette `0.585`, stability ARI `1.000`
  - mostly gapped cluster: 201 samples, median `R0 = 0.005347`
  - low-`U`/low-`V` cluster: 18 samples, median `R0 = 1.332`
- Gapped-region recommendations:
  - gap-open subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.734`, stability ARI `1.000`
  - strong-gap subset: raw `reG`, `|omega| <= 2.0`, `k=2`, silhouette `0.578`, stability ARI `1.000`, consensus ARI `0.825`
  - deep-gap subset: raw `reG`, `|omega| <= 0.25`, `k=2`, silhouette `0.619`, stability ARI `1.000`, consensus ARI `0.919`

### Kernel/SVM refinement
- Best overall kernel metric:
  - gap-open subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=4`, gamma factor `0.5`
  - silhouette `0.731`, bootstrap ARI `0.925`
- Full `V >= 0`, combined features, spectral clustering, `k=2`:
  - silhouette `0.662`, bootstrap ARI `0.948`
- Strong-gap subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=2`:
  - silhouette `0.575`, bootstrap ARI `0.969`

### Additional data recommendations
- 36 new suggestions were written to `create_code/UVset_additional_nonnegative_20260517.txt`.
- Top candidates:
  - `(4,1.454)`, `(4,1.172)`, `(4.094,1.454)`, `(8.781,2.25)`, `(3,1.289)`, `(4.875,2.351)`, `(4.406,1.5)`, `(9.843,2.5)`, `(9.907,0)`, `(2.593,1.227)`
- The recommended points are now mainly local threshold refinements near `R0 ~ 0.01` and `R0 ~ 0.10`, with a smaller number of strong-gap/deep-gap internal boundary checks.

## 2026-05-19 | Rerun after latest New suggested UVset data and skill creation

### Updated dataset
- Source combined data: `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- New ML-ready dataset: `data/Gcalpath/rho_re_g/ml_baseline_eta0p1_20260519/ml_ready_dataset.npz`
- Total sample count in combined file: 278
- `V >= 0` sample count: 229
  - `V > 0`: 200
  - `V = 0`: 29
- Only 10 of the 36 points from `create_code/UVset_additional_nonnegative_20260517.txt` are present in the newest combined data; 26 remain missing according to the 3-decimal `(U,V)` check.

### Fresh outputs
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p1_nonnegative_20260519`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p1_nonnegative_20260519`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260519`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p1_nonnegative_20260519`
- Kernel/SVM refinement: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p1_nonnegative_20260519`
- New suggested UV set: `create_code/UVset_additional_nonnegative_20260519.txt`

### Gap counts
- Gap-open, ratio `< 0.10`: 201 / 229
- Strong-gap, ratio `< 0.01`: 139 / 229
- Deep-gap, ratio `< 0.001`: 22 / 229
- Not gap-open: 28 / 229
- Median ratio: `0.006818`

### Supervised learning, five-fold CV
- `rho_G` ridge: `R2_U = 0.898`, `R2_V = 0.997`
- `rho_G` random forest: `R2_U = 0.957`, `R2_V = 0.941`
- `reG` random forest: `R2_U = 0.948`, `R2_V = 0.910`
- combined random forest: `R2_U = 0.960`, `R2_V = 0.927`

### Luttinger-path learning
- Strong-gap subset: 139 samples, all with two `reG(k,0)=0` path crossings.
- Ridge on Luttinger-path features:
  - Holdout: `R2_U = 0.976`, `R2_V = 0.986`
  - 5-fold CV: `R2_U = 0.765`, `R2_V = 0.967`

### Unsupervised learning
- Full `V >= 0` combined-feature KMeans:
  - `k=2`, silhouette `0.589`, stability ARI `1.000`
  - mostly gapped cluster: 211 samples, median `R0 = 0.005995`
  - low-`U`/low-`V` cluster: 18 samples, median `R0 = 1.332`
- Gapped-region recommendations:
  - gap-open subset: raw `rho_G`, `|omega| <= 2.0`, `k=2`, silhouette `0.728`, stability ARI `1.000`, consensus ARI `1.000`
  - strong-gap subset: raw `reG`, `|omega| <= 2.0`, `k=2`, silhouette `0.586`, stability ARI `1.000`, consensus ARI `0.829`
  - deep-gap subset: raw `reG`, `|omega| <= 0.25`, `k=2`, silhouette `0.618`, stability ARI `1.000`, consensus ARI `0.919`

### Kernel/SVM refinement
- Best overall kernel metric:
  - gap-open subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=4`, gamma factor `0.5`
  - silhouette `0.748`, bootstrap ARI `0.943`
- Full `V >= 0`, combined features, spectral clustering, `k=2`:
  - silhouette `0.668`, bootstrap ARI `0.947`
- Strong-gap subset, raw `rho_G`, `|omega| <= 2.0`, spectral clustering, `k=2`:
  - silhouette `0.571`, bootstrap ARI `0.977`

### Additional data recommendations
- 36 new suggestions were written to `create_code/UVset_additional_nonnegative_20260519.txt`.
- Top candidates:
  - `(4,1.446)`, `(4,1.164)`, `(3,1.3)`, `(4.812,2.363)`, `(4.39,1.5)`, `(8.765,2.25)`, `(9.828,2.5)`, `(9.922,0)`, `(2.577,1.231)`, `(4.11,1.446)`
- The new recommendations remain mostly fine threshold refinements near `R0 ~ 0.01` and `R0 ~ 0.10`, plus a few strong-gap/deep-gap cluster-boundary checks.

### Codex skill
- Created and validated reusable skill: `/Users/cosdis/.codex/skills/hk-green-ml-learning`
- Skill purpose: repeat this eta=0.1 `V >= 0` rho_G/reG supervised, Luttinger, unsupervised, kernel/SVM, UVset recommendation, and logging workflow in future sessions.

## 2026-05-19 | Eta broadening comparison (`eta = 0.02, 0.05, 0.10, 0.20`)

### Extraction and common sample set
- Re-extracted the same raw Lanczos sample set at three additional broadenings:
  - `eta = 0.20`: `data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p2_20260519.npz`
  - `eta = 0.05`: `data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p05_20260519.npz`
  - `eta = 0.02`: `data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p02_20260519.npz`
- Reused the current canonical `eta = 0.10` source:
  - `data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz`
- All four extractions contain 33,638 path spectra, 278 complete `(U,V)` groups, 1201 omega points over `[-15,15]`, and the same `V >= 0` learning set:
  - total samples: 278
  - `V >= 0`: 229
  - `V > 0`: 200
  - `V = 0`: 29

### Eta=0.02 workflow outputs
- Baseline ML-ready data: `data/Gcalpath/rho_re_g/ml_baseline_eta0p02_20260519`
- Gap analysis: `data/Gcalpath/rho_re_g/gap_analysis_eta0p02_nonnegative_20260519`
- Supervised physics features: `data/Gcalpath/rho_re_g/ml_physics_features_eta0p02_nonnegative_20260519`
- Strong-gap Luttinger-path learning: `data/Gcalpath/rho_re_g/ml_luttinger_path_reG0_eta0p02_nonnegative_gaplt0p01_20260519`
- Unsupervised learning: `data/Gcalpath/rho_re_g/ml_unsupervised_eta0p02_nonnegative_20260519`
- Kernel/SVM refinement: `data/Gcalpath/rho_re_g/ml_unsupervised_kernel_eta0p02_nonnegative_20260519`

### Updated eta-comparison outputs
- Comparison script: `create_code/compare_eta_learning.py`
- Comparison directory: `data/Gcalpath/rho_re_g/eta_comparison_20260519`
- Main files:
  - `summary.json`
  - `eta_gap_counts.csv`
  - `eta_gap_ratios_aligned.csv`
  - `eta_supervised_metrics.csv`
  - `eta_luttinger_metrics.csv`
  - `eta_unsupervised_metrics.csv`
  - `eta_kernel_metrics.csv`
  - `eta_pairwise_agreement.csv`
  - `eta_state_transitions.csv`
  - plots under `plots/`

### Gap analysis versus eta
- The gap proxy `R0 = max_k rho_G(k,0) / max_omega <rho_G>_path` is strongly broadening dependent.
- As `eta` decreases, `R0` decreases and more points move into strong/deep-gap bins:
  - `eta = 0.02`: gap-open 217 / 229, strong-gap 199 / 229, deep-gap 156 / 229, median `R0 = 0.000586`
  - `eta = 0.05`: gap-open 211 / 229, strong-gap 183 / 229, deep-gap 94 / 229, median `R0 = 0.002343`
  - `eta = 0.10`: gap-open 201 / 229, strong-gap 139 / 229, deep-gap 22 / 229, median `R0 = 0.006818`
  - `eta = 0.20`: gap-open 189 / 229, strong-gap 93 / 229, deep-gap 0 / 229, median `R0 = 0.021015`
- Pairwise rank ordering of `R0` is still almost unchanged even after adding `eta = 0.02`:
  - Spearman correlations range from `0.991` to `0.999` across all eta pairs.
- Threshold labels do change substantially:
  - from `eta = 0.02` to `eta = 0.05`: 6 gap-open labels, 16 strong-gap labels, and 62 deep-gap labels change.
  - from `eta = 0.05` to `eta = 0.10`: 10 gap-open labels, 44 strong-gap labels, and 72 deep-gap labels change.
  - from `eta = 0.10` to `eta = 0.20`: 12 gap-open labels, 46 strong-gap labels, and 22 deep-gap labels change.

### Supervised learning versus eta
- Random-forest supervised learning is broadly robust across eta, but `eta = 0.02` is modestly worse for `U`.
- Combined-feature random forest, 5-fold CV:
  - `eta = 0.02`: `R2_U = 0.950`, `R2_V = 0.923`
  - `eta = 0.05`: `R2_U = 0.960`, `R2_V = 0.922`
  - `eta = 0.10`: `R2_U = 0.960`, `R2_V = 0.927`
  - `eta = 0.20`: `R2_U = 0.962`, `R2_V = 0.932`
- `rho_G` random forest remains usable but is also weaker at `eta = 0.02`:
  - `eta = 0.02`: `R2_U = 0.938`, `R2_V = 0.916`
  - `eta = 0.05`: `R2_U = 0.956`, `R2_V = 0.937`
  - `eta = 0.10`: `R2_U = 0.957`, `R2_V = 0.941`
  - `eta = 0.20`: `R2_U = 0.965`, `R2_V = 0.944`
- Linear ridge is eta-sensitive for `reG`/combined features:
  - at `eta = 0.02`, combined ridge is stable and good: `R2_U = 0.898`, `R2_V = 0.998`.
  - at `eta = 0.20`, combined ridge is excellent: `R2_U = 0.969`, `R2_V = 0.998`.
  - at `eta = 0.10` and `0.05`, combined/reG ridge remains numerically unstable, while `rho_G` ridge remains usable.

### Luttinger-path learning versus eta
- The strong-gap subset size changes by eta:
  - `eta = 0.02`: 199 samples
  - `eta = 0.05`: 183 samples
  - `eta = 0.10`: 139 samples
  - `eta = 0.20`: 93 samples
- In all four cases, every selected strong-gap sample has two `reG(k,0)=0` path crossings.
- Ridge learning from Luttinger-path crossing features is weakest for the broad, less selective strong-gap subsets at small eta and strongest for the strict `eta = 0.20` subset:
  - `eta = 0.02`: 5-fold CV `R2_U = 0.211`, `R2_V = 0.818`
  - `eta = 0.05`: 5-fold CV `R2_U = 0.277`, `R2_V = 0.923`
  - `eta = 0.10`: 5-fold CV `R2_U = 0.765`, `R2_V = 0.967`
  - `eta = 0.20`: 5-fold CV `R2_U = 0.966`, `R2_V = 0.988`

### Unsupervised learning versus eta
- The global `V >= 0` KMeans result changes noticeably at `eta = 0.02`:
  - `eta = 0.02`: best KMeans `k=4`, silhouette `0.451`, stability ARI `0.996`
  - `eta = 0.05`: best KMeans `k=2`, silhouette `0.583`, stability ARI `0.995`
  - `eta = 0.10`: best KMeans `k=2`, silhouette `0.589`, stability ARI `1.000`
  - `eta = 0.20`: best KMeans `k=2`, silhouette `0.574`, stability ARI `0.952`
- Primary KMeans labels are identical between `eta = 0.05` and `eta = 0.10` (ARI `1.000`) and very similar between `eta = 0.10` and `eta = 0.20` (ARI `0.933`), but `eta = 0.02` has low primary-label ARI with the others (`~0.23--0.26`) because the optimal global cluster count changes from 2 to 4.
- Kernel/spectral best gapped-region metrics remain high:
  - `eta = 0.02`: strong-gap raw `rho_G`, spectral clustering `k=2`, silhouette `0.745`, bootstrap ARI `0.990`
  - `eta = 0.05`: gap-open raw `rho_G`, spectral clustering `k=4`, silhouette `0.711`, bootstrap ARI `0.922`
  - `eta = 0.10`: gap-open raw `rho_G`, spectral clustering `k=4`, silhouette `0.748`, bootstrap ARI `0.943`
  - `eta = 0.20`: gap-open raw `rho_G`, spectral clustering `k=3`, silhouette `0.734`, bootstrap ARI `0.929`

### Main conclusion
- The rank ordering of the gap proxy and the supervised random-forest learnability remain robust across eta.
- The exact threshold-based gap classification is not eta invariant: `eta = 0.02` resolves many more deep/strong gaps, while `eta = 0.20` broadens zero-frequency weight enough to eliminate the deep-gap class under the current `R0 < 0.001` criterion.
- The new `eta = 0.02` pass indicates that very sharp spectra start to expose additional substructure in unsupervised global clustering (`k=4` instead of `k=2`) and reduce the simplicity of Luttinger-only regression.
- For manuscript-style physics claims, keep `eta = 0.10` as the canonical middle broadening and use `eta = 0.02`, `0.05`, and `0.20` as systematic-broadening checks rather than mixing their threshold labels directly.

### Data-recommendation preference
- User preference recorded: future additional-data suggestions should treat `eta = 0.10` and `eta = 0.05` as the most important and equally weighted broadenings.
- Use `eta = 0.02` and `eta = 0.20` only as secondary robustness checks unless explicitly requested otherwise.
- Practical rule for future UVset recommendations:
  - prioritize points selected by either `eta = 0.10` or `eta = 0.05`;
  - give extra priority to regions that are physically consistent between the two even if exact 3-decimal suggestions differ;
  - include `eta = 0.02`/`0.20` candidates only when they reinforce the `eta = 0.10`/`0.05` physics or expose a useful boundary not covered by the co-primary eta values.

## 2026-05-21 | CADES refresh and co-primary eta workflow (`eta = 0.10, 0.05`)

### Raw-data refresh
- Refreshed CADES `L*.dat` files from `/home/9pm/ED_qbasis/qbasis/bin/ML/HF/Gcalpath` into `data/Gcalpath/Lanczos_raw_data`.
- Remote inventory: 4314 files, `270M	total`; filename conflicts: 0.
- Local raw file count: 66720 -> 71034; remote files missing locally after transfer: 0.

### rho_G/reG regeneration
- Regenerated both co-primary broadenings in parallel:
  - `eta = 0.10`: `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p1_20260521.npz`
  - `eta = 0.05`: `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p05_20260521.npz`
- Both branches accepted 293 complete 121-point `(U,V,tp)` groups (35453 path spectra) and used 1201 omega points.
- Three groups were still incomplete and skipped in both branches: `(U,V,tp)=(4,1.312,0)`, `(9.5,3.312,0)`, `(10,2.593,0)`.

### Nonnegative-V learning sample set
- Total samples: 293; `V >= 0`: 244; `V > 0`: 215; `V = 0`: 29.
- Previous co-primary recommendation coverage from `create_code/UVset_additional_coprimary_eta0p1_eta0p05_20260519.txt`: 15 / 36 present in the complete regenerated datasets.

### Gap proxy counts (`R0 = max_k rho_G(k,0) / max_omega <rho_G>_path`)
- `eta = 0.10`: gap-open 216 / 244, strong-gap 151, deep-gap 23, median `R0=0.00733096`.
- `eta = 0.05`: gap-open 226 / 244, strong-gap 196, deep-gap 97, median `R0=0.00243749`.

### Supervised and Luttinger learning
- `eta = 0.10` five-fold CV:
  - rho ridge `R2_U=0.906`, `R2_V=0.997`; rho RF `R2_U=0.965`, `R2_V=0.944`; reG RF `R2_U=0.956`, `R2_V=0.908`; combined RF `R2_U=0.965`, `R2_V=0.929`.
  - Luttinger subset: 151 samples, crossings {'2': 151}, ridge CV `R2_U=0.842`, `R2_V=0.964`.
- `eta = 0.05` five-fold CV:
  - rho ridge `R2_U=0.916`, `R2_V=0.998`; rho RF `R2_U=0.960`, `R2_V=0.942`; reG RF `R2_U=0.953`, `R2_V=0.908`; combined RF `R2_U=0.965`, `R2_V=0.923`.
  - Luttinger subset: 196 samples, crossings {'2': 196}, ridge CV `R2_U=0.347`, `R2_V=0.871`.

### Unsupervised and kernel/SVM refinement
- `eta = 0.10`: primary KMeans `kmeans_k2`, silhouette `0.598`, stability ARI `1.000`; best kernel metric `nonnegative_v_gap_open / raw_rho_low_w2p0_pca / spectral_clustering` with silhouette `0.753` and bootstrap ARI `0.902`.
- `eta = 0.05`: primary KMeans `kmeans_k2`, silhouette `0.592`, stability ARI `0.995`; best kernel metric `nonnegative_v_gap_open / raw_rho_low_w2p0_pca / spectral_clustering` with silhouette `0.720` and bootstrap ARI `0.908`.

### Recommendation files
- Eta-specific recommendation files:
  - `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/create_code/UVset_additional_eta0p1_nonnegative_20260521.txt`
  - `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/create_code/UVset_additional_eta0p05_nonnegative_20260521.txt`
- Co-primary recommendation file: `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/create_code/UVset_additional_coprimary_eta0p1_eta0p05_20260521.txt`
- Co-primary top candidates: (10.500,2.969), (7.812,2.617), (4.000,1.442), (10.000,2.938), (8.500,2.173), (9.500,2.812).
- Machine-readable run summary: `data/Gcalpath/rho_re_g/pipeline_summary_eta0p1_eta0p05_20260521.json`.



## 2026-05-21 | CADES completion refresh and co-primary eta rerun (`eta = 0.10, 0.05`, tag `20260521b`)

### Raw-data refresh
- Downloaded the 6 newly generated CADES remote-only `L*.dat` files from `/home/9pm/ED_qbasis/qbasis/bin/ML/HF/Gcalpath` into `data/Gcalpath/Lanczos_raw_data` without overwriting existing local data.
- Current CADES inventory: 4320 files; local raw file count: 71034 -> 71040; remote files missing locally after verification: 0.
- The downloaded files completed the three previously incomplete positive-V groups: `(U,V)=(10,2.593)`, `(9.5,3.312)`, and `(4,1.312)`.

### rho_G/reG regeneration
- Regenerated both co-primary broadenings in parallel:
  - `eta = 0.10`: `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p1_20260521b.npz`
  - `eta = 0.05`: `/Users/cosdis/Desktop/projects/HK_orbital_cluster/ML_MMHK/data/Gcalpath/rho_re_g/rho_re_g_all_complete_eta0p05_20260521b.npz`
- Both branches accepted 296 complete 121-point `(U,V,tp)` groups, skipped 0 incomplete groups, and saved 35816 path spectra with 1201 omega points.
- The backward-compatible `rho_re_g_all_complete.npz` was updated to the `eta = 0.10` branch.

### Nonnegative-V learning sample set
- Total samples: 296; `V >= 0`: 247; `V > 0`: 218; `V = 0`: 29.
- Coverage of `create_code/UVset_additional_coprimary_eta0p1_eta0p05_20260519.txt` improved from 15 / 36 to 18 / 36 present in the complete regenerated datasets.
- Newly present old recommendations: `(10.000,2.593)`, `(9.500,3.312)`, `(4.000,1.312)`.

### Gap proxy counts (`R0 = max_k rho_G(k,0) / max_omega <rho_G>_path`)
- `eta = 0.10`: gap-open 219 / 247, strong-gap 153, deep-gap 23, median `R0=0.0072814856`.
- `eta = 0.05`: gap-open 229 / 247, strong-gap 199, deep-gap 98, median `R0=0.0024323926`.

### Learning highlights
- `eta = 0.10`: rho ridge CV `R2_U=0.903`, `R2_V=0.997`; combined RF CV `R2_U=0.964`, `R2_V=0.927`; Luttinger subset 153 samples with crossings {'2': 153}.
- `eta = 0.05`: rho ridge CV `R2_U=0.920`, `R2_V=0.998`; combined RF CV `R2_U=0.963`, `R2_V=0.922`; Luttinger subset 199 samples with crossings {'2': 199}.
- Kernel/SVM best metrics remain gap-open raw-`rho_G` spectral clustering: `eta = 0.10` silhouette 0.742, bootstrap ARI 0.911; `eta = 0.05` silhouette 0.722, bootstrap ARI 0.959.

### Recommendation files
- Eta-specific recommendation files:
  - `create_code/UVset_additional_eta0p1_nonnegative_20260521b.txt`
  - `create_code/UVset_additional_eta0p05_nonnegative_20260521b.txt`
- Co-primary recommendation files:
  - `create_code/UVset_additional_coprimary_eta0p1_eta0p05_20260521b.txt`
  - `create_code/UVset_additional_coprimary_eta0p1_eta0p05_20260521b.csv`
- Co-primary top candidates: (4.000,1.442), (10.500,2.969), (4.000,1.160), (8.500,2.173), (3.000,1.294), (4.000,1.281).
- Machine-readable run summary: `data/Gcalpath/rho_re_g/pipeline_summary_eta0p1_eta0p05_20260521b.json`.
