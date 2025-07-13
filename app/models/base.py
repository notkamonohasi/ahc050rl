from torch import Tensor, nn

from app.exceptions.utils import NotImplementedError


class BaseRLModel(nn.Module):
    def forward(self, rocks: Tensor, probs: Tensor) -> Tensor:
        """
        (N, N) を入力したときは (N, N) を出力する
        (B, N, N) を入力したときは (B, N, N) を出力する
        """
        raise NotImplementedError
