
# TRIP PyTorch Example

This repository contains:

- `trip.py`  
  TRIP training and prediction code.

- `trip_synthetic.py`  
  Synthetic graph generator based on the TRIP generative story.

- `evaluate.py`  
  End-to-end example that:
  1. generates synthetic graphs,
  2. trains TRIP,
  3. evaluates prediction accuracy and subspace recovery.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python evaluate.py
```

## What `python evaluate.py` does

Running `python evaluate.py` performs the following steps:

1. Generates a synthetic dataset of graph matrices and targets.
2. Splits the data into train and test subsets.
3. Trains the TRIP model on the training graphs.
4. Predicts the target values for the test graphs.
5. Reports:
   - MSE
   - RMSE
   - MAE
   - Pearson correlation
   - R²
   - subspace recovery score

Example output:

```text
Generating synthetic graphs...
Training TRIP...
epoch=   1 loss=7.363878 pred=7.363824 recon=0.053434
epoch=  50 loss=0.259815 pred=0.259772 recon=0.042934
epoch= 100 loss=0.093093 pred=0.093052 recon=0.041355
epoch= 150 loss=0.057096 pred=0.057055 recon=0.041250
epoch= 200 loss=0.037196 pred=0.037155 recon=0.041549

=== Evaluation ===
     mse: 0.209120
    rmse: 0.457297
     mae: 0.293039
    corr: 0.982549
      r2: 0.963570
subspace: 1.805619
```

## Overview

The training code learns an orthonormal projection matrix `C` and a predictor `H` such that

```text
U_k = C^T W_k C
c_k ≈ H(U_k)
```

while also encouraging reconstruction of each graph:

```text
W_k ≈ C U_k C^T
```

The synthetic generator creates graphs by:

1. sampling mechanism-specific orthonormal matrices `C_s`,
2. sampling mechanism-specific neural mappings `H_s`,
3. sampling target indicators `y_s`,
4. optimizing a graph matrix `W` so that:
   - `H_s(C_s^T W C_s)` matches `y_s`,
   - `W` remains reconstructable from the latent mechanisms,
   - `W` is encouraged to be acyclic through a NOTEARS-style smooth penalty.

## Files

### `trip.py`
Defines:
- `Predictor`
- `orthonormalize`
- `trip_train`
- `trip_predict`

### `trip_synthetic.py`
Defines:
- `SyntheticTRIPConfig`
- `generate_trip_synthetic_dataset`
- `principal_angle_similarity`

### `evaluate.py`
Runs the complete experiment pipeline and prints evaluation metrics.

## Notes

- The DAG constraint is implemented with a smooth NOTEARS-style acyclicity penalty.
- PyTorch autograd is used throughout.
- The current configuration is chosen so that `python evaluate.py` runs in a reasonable amount of time on CPU.

## Suggested next steps

Natural extensions include:
- multiple random seeds,
- shuffled train/test splits,
- cross-validation,
- stronger augmented-Lagrangian optimization for DAG generation,
- multi-target prediction,
- integration with a full NOTEARS implementation.

## License

MIT
