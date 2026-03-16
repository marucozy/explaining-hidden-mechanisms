
from __future__ import annotations

from typing import List, Tuple

import torch

from trip import TrainResult, trip_predict, trip_train
from trip_synthetic import (
    SyntheticTRIPConfig,
    generate_trip_synthetic_dataset,
    principal_angle_similarity,
)


def train_test_split_lists(
    W_list: List[torch.Tensor],
    c_list: List[torch.Tensor],
    test_ratio: float = 0.2,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Deterministic split without shuffling.
    """
    n = len(W_list)
    n_test = int(n * test_ratio)
    n_train = n - n_test

    W_train = W_list[:n_train]
    W_test = W_list[n_train:]
    c_train = c_list[:n_train]
    c_test = c_list[n_train:]

    return W_train, W_test, c_train, c_test


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    Compute standard regression metrics.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(y_true - y_pred)).item()

    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = torch.sqrt((yt ** 2).sum()) * torch.sqrt((yp ** 2).sum()) + 1e-12
    corr = ((yt * yp).sum() / denom).item()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
    r2 = (1.0 - ss_res / ss_tot).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "r2": r2,
    }


def main() -> None:
    config = SyntheticTRIPConfig()

    print("Generating synthetic graphs...")
    data = generate_trip_synthetic_dataset(config)

    W_list = data["W_list"]
    c_list = data["c_list"]
    C_true = data["C_true_list"][0]

    W_train, W_test, c_train, c_test = train_test_split_lists(
        W_list=W_list,
        c_list=c_list,
        test_ratio=0.2,
    )

    print("Training TRIP...")
    result: TrainResult = trip_train(
        W_list=W_train,
        c_list=c_train,
        latent_dim=config.latent_dim,
        beta=1e-3,
        lr=1e-2,
        max_epoch=200,
        hidden_dim=32,
        verbose=True,
        device="cpu",
    )

    pred_test = trip_predict(
        C=result.C,
        predictor=result.predictor,
        W_list=W_test,
    )
    y_test = torch.stack(c_test, dim=0)

    metrics = regression_metrics(y_true=y_test, y_pred=pred_test)
    subspace_score = principal_angle_similarity(C_true, result.C)

    print("\n=== Evaluation ===")
    for key, value in metrics.items():
        print(f"{key:>8s}: {value:.6f}")

    print(f"{'subspace':>8s}: {subspace_score:.6f}")
    print("\nInterpretation:")
    print("- Smaller mse/rmse/mae is better.")
    print("- Larger corr/r2 is better.")
    print("- For latent_dim = 2, the maximum subspace score is 2.0.")


if __name__ == "__main__":
    main()
