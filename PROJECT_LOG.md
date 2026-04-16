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
