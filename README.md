# Rewiring Neural Developmental Programs

Code and results accompanying “Building Neural Policies Through Developmental Rewiring”.

## Methods

The experiments compare:
- Neural Developmental Programs (NDPs).
- Rewiring Neural Developmental Programs (RNDPs).
- Direct encoding (DE), using two hidden layers of 16 neurons.

Each encoding is evaluated with and without neuron-centric Hebbian learning.

## Installation

Run commands from the repository root in a Python environment.

SWIG is already listed in `requirements.txt`. Install it first so it is available if the Box2D Python bindings need to be compiled for LunarLander. If SWIG is already available, skip the first command.

```bash
python -m pip install swig
python -m pip install -r requirements.txt
python -m pip install pandas scipy networkx popgym
```

## Included results

Each task and model condition has 30 independent evolutionary runs.

- `Results/september2026_paper_final_ndp_rndp32/`: consolidated NDP and RNDP experiments, saved models, testing evaluations and graph statistics.
- `Results/october2026_fixed_ann_16_16_pilot/`: DE experiments and testing evaluations. Despite the historical folder name, these are the 16–16 baselines used in the paper.
- `Results/fixed_ann_ndp_rndp_comparison/`: combined training/testing fitness, target-attainment counts, Figure 1 and statistical comparisons.

The `posthoc_test_evaluation` directories contain held-out evaluations.

## Recreate manuscript figures

### Figure 1: training and testing fitness

```bash
python Utilities/plot_normalised_fitness.py \
  Results/fixed_ann_ndp_rndp_comparison/combined_fitness_results.csv \
  --raw
```

### Figure 2: developed network sizes

```bash
python Utilities/plot_graph_size_heatmaps.py \
  Results/september2026_paper_final_ndp_rndp32/figures/matched_run_metrics.csv
```

### Figure 3: node-state diversity

```bash
python Utilities/plot_graph_summary_dotplots.py \
  Results/september2026_paper_final_ndp_rndp32/figures/matched_run_metrics.csv \
  --diversity-only
```

This command generates diversity plots for both all nodes and functional nodes. The manuscript uses the functional-node plot.

## Statistical comparisons

```bash
python Utilities/test_fitness_permutations.py
```

The analysis uses two-sided, unpaired permutation tests of differences in mean testing fitness, with 99,999 random permutations. Holm correction is applied separately to the two families of 15 comparisons. Settings and outputs are stored in `Results/fixed_ann_ndp_rndp_comparison/permutation_tests/`.

## Running experiments

Inspect the available options and resolved configuration before launching experiments:

```bash
python Experiments/final_experiments.py --help

python Experiments/final_experiments.py \
  --tasks cartpole \
  --models standard_ndp rewiring_ndp fixed_mlp \
  --policy-hebbian all \
  --seeds 0 \
  --algorithm EA \
  --output Results/reproduction \
  --dry-run
```

Task configurations are defined in `Tasks/`. Retained experiment manifests document the saved runs. Remove `--dry-run` to execute the selected experiment conditions.

## Code organisation

- `NDP/`: developmental encodings and policy networks.
- `Baseline/`: directly encoded policy networks.
- `Graph/`: graph representations.
- `Optimisation/`: optimisation algorithms.
- `Tasks/`: environments and task configurations.
- `Experiments/`: experiment launchers.
- `Utilities/`: evaluation, analysis and plotting scripts.
