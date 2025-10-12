import torch
from torch import Tensor
import math
import numpy as np


def to_tensor(x, device=None, dtype=torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def angle_wrap(theta: Tensor) -> Tensor:
    # wrap to [-pi, pi] MKEHDM
    return (theta + math.pi) % (2 * math.pi) - math.pi

def var_ru(samples: Tensor, beta: float) -> Tensor:
    """
    Value at Risk (VaR) at confidence level beta over 1D samples.
    samples: (..., N)
    returns: (...) scalar VaR
    """
    assert 0.0 < beta < 1.0
    return torch.quantile(samples, q=beta, dim=-1)

def var_lower_tail(values: Tensor, alpha: float) -> Tensor:
    """
    Lower-tail VaR of 1D values (smaller is worse).
    values: (..., N)
    returns: (...) VaR over the lower tail
    """
    # lower-tail VaR: same as upper-tail VaR of (-values)
    return -var_ru(-values, beta=1.0 - alpha)

def cvar_ru(samples: Tensor, beta: float) -> Tensor:
    """
    Rockafellar–Uryasev CVaR_{1-beta} over 1D samples (larger is worse if you're using "loss").
    samples: (..., N)
    returns: (...) scalar CVaR
    """
    assert 0.0 < beta < 1.0
    q = torch.quantile(samples, q=beta, dim=-1)
    tail = (1.0 - beta)
    # RU formula: CVaR = t + 1/(1-beta) * E[(X - t)+]
    # broadcast q
    q_exp = q.unsqueeze(-1)
    excess = torch.relu(samples - q_exp).mean(dim=-1)
    return q + excess / max(1e-6, tail)

def cvar_lower_tail(values: Tensor, alpha: float) -> Tensor:
    """
    Lower-tail CVaR of 1D values (smaller is worse).
    values: (..., N)
    returns: (...) CVaR over the lower tail
    """
    # lower-tail of m: same as upper-tail of (-m)
    return -cvar_ru(-values, beta=1.0 - alpha)

def softplus(z: Tensor, beta: float = 30.0) -> Tensor:
    # numerically stable softplus with tunable sharpness
    return torch.log1p(torch.exp(beta * z)) / beta

def metrics_min_distance(traj_ego, traj_obs):
    """
    Return the minimum inter-agent distance between ego and obstacle trajectories.

    traj_ego: (T, 2/3/...) array-like  [x, y, (theta, ...)]
    traj_obs: (T, 2/3/...) array-like  [x, y, (theta, ...)]
    """
    # to numpy
    ego = np.asarray(traj_ego)
    obs = np.asarray(traj_obs)

    # match lengths (use the overlapping prefix)
    T = min(len(ego), len(obs))
    if T == 0:
        raise ValueError("Empty trajectories.")

    # use XY only
    ego_xy = ego[:T, :2]
    obs_xy = obs[:T, :2]

    # per-step distances and minimum
    dists = np.linalg.norm(ego_xy - obs_xy, axis=1)
    d_min = float(dists.min())

    # optional: index where min occurs
    # idx_min = int(dists.argmin())

    print(f"[metrics] min ego–obs distance: {d_min:.3f}")
    return d_min