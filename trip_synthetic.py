
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IndicatorMLP(nn.Module):
    """
    Mechanism-specific mapping H_s[U_s] -> y_s.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 10, out_dim: int = 3):
        super().__init__()
        in_dim = latent_dim * latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        x = U.reshape(U.shape[0], -1)
        return self.net(x)


@dataclass
class SyntheticTRIPConfig:
    n_graphs: int = 60
    n_observed: int = 10
    n_mechanisms: int = 2
    latent_dim: int = 2
    n_indicators: int = 3
    hidden_dim: int = 10
    beta: float = 1e-3
    lambda_l1: float = 0.01
    rho_dag: float = 10.0
    dag_steps: int = 80
    dag_lr: float = 1e-2
    seed: int = 0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def orthonormal_matrix(
    n_rows: int,
    n_cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Sample a random orthonormal matrix by SVD.
    """
    Z = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    P, _, Vh = torch.linalg.svd(Z, full_matrices=False)
    return P @ Vh


def notears_acyclicity(W: torch.Tensor) -> torch.Tensor:
    """
    Smooth acyclicity function from NOTEARS:
        h(W) = tr(exp(W ∘ W)) - d
    """
    d = W.shape[0]
    return torch.trace(torch.matrix_exp(W * W)) - d


def trip_generation_objective(
    W: torch.Tensor,
    C_list: Sequence[torch.Tensor],
    H_list: Sequence[nn.Module],
    y_target_list: Sequence[torch.Tensor],
    beta: float,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    TRIP-inspired generation objective for one graph:
        D(W) = sum_s loss(y_s, H_s(U_s)) + beta ||W - sum_s C_s U_s C_s^T||_F^2
        U_s = C_s^T W C_s
    """
    pred_loss = torch.tensor(0.0, device=W.device)
    recon_terms: List[torch.Tensor] = []
    U_list: List[torch.Tensor] = []
    y_pred_list: List[torch.Tensor] = []

    for C_s, H_s, y_s in zip(C_list, H_list, y_target_list):
        U_s = C_s.T @ W @ C_s
        y_pred_s = H_s(U_s.unsqueeze(0)).squeeze(0)

        pred_loss = pred_loss + F.mse_loss(y_pred_s, y_s)
        recon_terms.append(C_s @ U_s @ C_s.T)
        U_list.append(U_s)
        y_pred_list.append(y_pred_s)

    recon = torch.stack(recon_terms, dim=0).sum(dim=0)
    recon_loss = ((W - recon) ** 2).mean()

    D = pred_loss + beta * recon_loss
    return D, U_list, y_pred_list


def optimize_single_W(
    C_list: Sequence[torch.Tensor],
    H_list: Sequence[nn.Module],
    y_target_list: Sequence[torch.Tensor],
    beta: float,
    lambda_l1: float,
    rho_dag: float,
    dag_steps: int,
    dag_lr: float,
) -> torch.Tensor:
    """
    Optimize one observed graph matrix W under a smooth DAG penalty.
    """
    device = C_list[0].device
    n_observed = C_list[0].shape[0]

    W = nn.Parameter(0.01 * torch.randn(n_observed, n_observed, device=device))
    optimizer = torch.optim.Adam([W], lr=dag_lr)

    for _ in range(dag_steps):
        optimizer.zero_grad()

        W_masked = W - torch.diag(torch.diag(W))

        D, _, _ = trip_generation_objective(
            W=W_masked,
            C_list=C_list,
            H_list=H_list,
            y_target_list=y_target_list,
            beta=beta,
        )

        h = notears_acyclicity(W_masked)
        l1 = torch.abs(W_masked).mean()

        loss = D + lambda_l1 * l1 + 0.5 * rho_dag * h * h
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_final = W - torch.diag(torch.diag(W))

    return W_final.detach()


def principal_angle_similarity(C_true: torch.Tensor, C_est: torch.Tensor) -> float:
    """
    Compare two subspaces using the singular values of C_true^T C_est.
    For latent_dim = J, the maximum value is J.
    """
    M = C_true.T @ C_est
    s = torch.linalg.svdvals(M)
    return float(s.sum().item())


def generate_trip_synthetic_dataset(
    config: SyntheticTRIPConfig,
) -> Dict[str, object]:
    """
    Generate a synthetic dataset for TRIP experiments.
    """
    set_seed(config.seed)

    device = torch.device(config.device)
    dtype = torch.float32

    C_true_list = [
        orthonormal_matrix(config.n_observed, config.latent_dim, device, dtype)
        for _ in range(config.n_mechanisms)
    ]

    H_list = [
        IndicatorMLP(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.n_indicators,
        ).to(device)
        for _ in range(config.n_mechanisms)
    ]

    for H_s in H_list:
        for p in H_s.parameters():
            with torch.no_grad():
                p.copy_(torch.randn_like(p))

    W_list: List[torch.Tensor] = []
    c_list: List[torch.Tensor] = []
    full_y_list: List[torch.Tensor] = []

    for _ in range(config.n_graphs):
        y_target_list = [
            -5.0 + 10.0 * torch.rand(config.n_indicators, device=device, dtype=dtype)
            for _ in range(config.n_mechanisms)
        ]

        W_k = optimize_single_W(
            C_list=C_true_list,
            H_list=H_list,
            y_target_list=y_target_list,
            beta=config.beta,
            lambda_l1=config.lambda_l1,
            rho_dag=config.rho_dag,
            dag_steps=config.dag_steps,
            dag_lr=config.dag_lr,
        )

        c_k = y_target_list[0][0].reshape(1)

        W_list.append(W_k.cpu())
        c_list.append(c_k.cpu())
        full_y_list.append(torch.cat(y_target_list, dim=0).cpu())

    return {
        "W_list": W_list,
        "c_list": c_list,
        "full_y_list": full_y_list,
        "C_true_list": [C.cpu() for C in C_true_list],
        "H_list": H_list,
    }
