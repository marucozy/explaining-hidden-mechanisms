
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """
    Predictor H[U] that maps a projected latent matrix U_k to a target c_k.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        in_dim = latent_dim * latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        x = U.reshape(U.shape[0], -1)
        return self.net(x)


def orthonormalize(Z: torch.Tensor) -> torch.Tensor:
    """
    Retract Z onto the Stiefel manifold via SVD:
        Z = P S Q^T -> C = P Q^T
    """
    P, _, Vh = torch.linalg.svd(Z, full_matrices=False)
    return P @ Vh


@dataclass
class TrainResult:
    C: torch.Tensor
    predictor: nn.Module
    history: Dict[str, List[float]]


def trip_train(
    W_list: List[torch.Tensor],
    c_list: List[torch.Tensor],
    latent_dim: int = 2,
    beta: float = 1e-3,
    lr: float = 1e-2,
    max_epoch: int = 300,
    hidden_dim: int = 32,
    verbose: bool = True,
    device: str = "cpu",
) -> TrainResult:
    if len(W_list) == 0 or len(c_list) == 0:
        raise ValueError("W_list and c_list must be non-empty.")
    if len(W_list) != len(c_list):
        raise ValueError("W_list and c_list must have the same length.")

    device_t = torch.device(device)
    dtype = torch.float32

    W = torch.stack([w.to(dtype=dtype) for w in W_list], dim=0).to(device_t)
    c = torch.stack([x.to(dtype=dtype) for x in c_list], dim=0).to(device_t)

    _, n_observed, _ = W.shape
    out_dim = c.shape[1]

    Z = nn.Parameter(torch.randn(n_observed, latent_dim, device=device_t, dtype=dtype) * 0.1)
    predictor = Predictor(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device_t)

    optimizer = torch.optim.Adam([Z] + list(predictor.parameters()), lr=lr)

    history = {"loss": [], "pred_loss": [], "recon_loss": []}

    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad()

        C = orthonormalize(Z)  # shape: (I, J)

        # U_k = C^T W_k C, shape: (K, J, J)
        U = torch.einsum("ab,kac,cd->kbd", C, W, C)

        pred = predictor(U)
        pred_loss = F.mse_loss(pred, c)

        # W_hat_k = C U_k C^T, shape: (K, I, I)
        W_hat = torch.einsum("ab,kbd,cd->kac", C, U, C)

        recon_loss = ((W - W_hat) ** 2).mean()

        loss = pred_loss + beta * recon_loss
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["pred_loss"].append(float(pred_loss.item()))
        history["recon_loss"].append(float(recon_loss.item()))

        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == max_epoch):
            print(
                f"epoch={epoch:4d} "
                f"loss={loss.item():.6f} "
                f"pred={pred_loss.item():.6f} "
                f"recon={recon_loss.item():.6f}"
            )

    with torch.no_grad():
        C_final = orthonormalize(Z).detach().cpu()

    return TrainResult(
        C=C_final,
        predictor=predictor.cpu(),
        history=history,
    )


@torch.no_grad()
def trip_predict(
    C: torch.Tensor,
    predictor: nn.Module,
    W_list: List[torch.Tensor],
) -> torch.Tensor:
    if len(W_list) == 0:
        raise ValueError("W_list must be non-empty.")

    dtype = torch.float32
    W = torch.stack([w.to(dtype=dtype) for w in W_list], dim=0)
    C = C.to(dtype=dtype)

    U = torch.einsum("ab,kac,cd->kbd", C, W, C)
    pred = predictor(U)
    return pred.cpu()
