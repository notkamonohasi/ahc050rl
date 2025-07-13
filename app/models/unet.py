import torch
import torch.nn as nn
from torch import Tensor

from app.const import N
from app.models import BaseRLModel


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNetModel(BaseRLModel):
    def __init__(self, channels: tuple[int, int, int], in_channels=2, out_channels=1):
        super().__init__()

        # エンコーダ
        self.enc1 = DoubleConv(in_channels, channels[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(channels[0], channels[1])
        self.pool2 = nn.MaxPool2d(2)

        # ボトルネック
        self.bottleneck = DoubleConv(channels[1], channels[2])

        # デコーダ
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1])

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0])

        # 出力
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, rocks: Tensor, probs: Tensor):
        assert rocks.size(dim=-1) == probs.size(dim=-1) == N
        assert rocks.size(dim=-2) == probs.size(dim=-2) == N
        assert rocks.size() == probs.size()
        assert rocks.dim() in [2, 3]

        x: Tensor = torch.cat([rocks.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)

        # PyTorchのConv2dは (batch, channels, height, width) の順序を期待
        # 現在は (height, width, channels) なので並び替える
        if rocks.dim() == 2:
            # (height, width, channels) -> (1, channels, height, width)
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif rocks.dim() == 3:
            # (batch, height, width, channels) -> (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Invalid dimension: {rocks.dim()}")

        # エンコーダ
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))

        # ボトルネック
        x3 = self.bottleneck(self.pool2(x2))

        # デコーダ
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.out_conv(x)

        if rocks.dim() == 2:
            x = x.view(N, N)
        elif rocks.dim() == 3:
            x = x.view(-1, N, N)
        else:
            raise ValueError(f"Invalid dimension: {rocks.dim()}")

        return x
