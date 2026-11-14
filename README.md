# INFOMDW Assignment 4 – Bias mitigation with Reweighing

This repository contains the code for Assignment 4 (Paper Review and Algorithm Evaluation – Algorithmic Fairness).

**Algorithm:** Reweighing (pre-processing), from F. Kamiran and T. Calders, "Data preprocessing techniques for classification without discrimination", *Knowledge and Information Systems* 33(1), 2012.
**Implementation:** IBM AI Fairness 360 (`aif360.algorithms.preprocessing.Reweighing`), cross-checked with a manual implementation of the weight formula.

## Contents

| File | Description |
|---|---|
| `assignment4_reweighing.ipynb` | Full mitigation pipeline for the three datasets, with outputs |
| `results/results_seed42.csv` | Fairness and performance measures of the main run (seed 42) |
| `results/results_repeated.csv` | Test-set results over 10 random splits (mean ± sd) |
| `results/before_after.png` | Figure comparing the measures before and after mitigation |

## Datasets

| Dataset | Source | Sensitive attribute (privileged group) | Favourable label |
|---|---|---|---|
| Adult | UCI, `adult.data` | sex (male) | income > 50K |
| German Credit | UCI, `german.data` | age (older than 25) | good credit risk |
| COMPAS | ProPublica, `compas-scores-two-years.csv` | race (Caucasian vs. African-American) | no recidivism within two years |

The raw files are downloaded automatically into `data/` the first time the notebook runs.

## How to run

Tested with Python 3.11.

```
conda create -n fairness python=3.11 -y
conda activate fairness
pip install aif360 scikit-learn pandas numpy matplotlib ipykernel
```

Open `assignment4_reweighing.ipynb`, select the `fairness` kernel and run all cells. Warnings about missing optional AIF360 dependencies (tensorflow, fairlearn, inFairness) can be ignored.

## Pipeline

1. Compute DP_Diff, DIR and consistency on the original labels of each full dataset.
2. 70/30 train/test split, stratified on the sensitive attribute and the label.
3. Compute the Reweighing weights on the training set and train a logistic regression with them; an unweighted logistic regression is the baseline. The sensitive attribute is not used as a feature (assumption A2 of the paper).
4. Compute DP_Diff, DIR, consistency, accuracy and balanced accuracy on the test set.
