import enum
import torch
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class ActionMode(enum.Enum):
    STUDENT = 0
    TEACHER = 1
    UNCERTAIN = 2


def energy_value(state: Any, p_mat: np.ndarray) -> int:
    """
    Get energy value represented by s^T @ P @ s -> return a value
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    return np.squeeze(np.asarray(state).T @ p_mat @ state)


def energy_value_2d(state: torch.Tensor, p_mat: torch.Tensor) -> torch.Tensor:
    """
    Get energy value represented by s^T @ P @ s (state is a 2d vector) -> return a 1d array
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    sp = torch.matmul(state, p_mat)

    return torch.sum(sp * state, dim=1)
