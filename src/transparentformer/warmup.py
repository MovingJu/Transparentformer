import torch


def batch_mat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: `(B, r1, c1)`, b: `(B, c1, c2)` -> `(B, r1, c2)`\n
    where
        `B`: batch size
        `r, c`: size of row, column
    """
    return torch.einsum("bij,bjk->bik", a, b)
