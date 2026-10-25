# Copyright 2026 Norma Core contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quarter checker: which pizza quarters got a new piece between two scene frames.

Input is a (before, after) pair of scene-camera frames, both with the arm parked.
Output is 4 logits, one per quarter in QUARTERS order: "a piece was added here".
Comparing two frames instead of judging one keeps it independent of whatever
is already on the pizza (printed toppings, earlier pieces, the other ingredient).

Training pairs come from the recorded episodes (scripts/train_checker.py):
start-of-episode vs end-of-episode frames are positives for that episode's
quarter; two frames from the same parked moment are negatives, and so is a
reversed pair (a piece can't be removed).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

QUARTERS = ("top-left", "top-right", "bottom-right", "bottom-left")
IMAGE_SIZE = 224
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def jpeg_to_tensor(jpeg: bytes) -> torch.Tensor:
    """JPEG bytes -> (3, 224, 224) float in [0, 1]."""
    with Image.open(io.BytesIO(jpeg)) as im:
        im = im.convert("RGB")
        if im.size != (IMAGE_SIZE, IMAGE_SIZE):
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


class QuarterChecker(nn.Module):
    """Siamese ResNet18 trunk (to layer3) + a small spatial head over [before, after, after-before]."""

    def __init__(self, pretrained: bool = True, crop: tuple[int, int, int, int] | None = None):
        super().__init__()
        # (x0, y0, x1, y1) box around the pizza in the 224x224 scene frame; keeps the
        # arm and people out of view, so only the toppings change between frames.
        self.crop = tuple(crop) if crop else None
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        for p in list(self.trunk.parameters())[: -len(list(r.layer3.parameters()))]:
            p.requires_grad = False  # train only layer3 of the trunk
        self.head = nn.Sequential(
            nn.Conv2d(256 * 3, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64 * 16, len(QUARTERS)),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] each -> (B, 4) logits."""
        mean, std = _MEAN.to(before.device), _STD.to(before.device)
        if self.crop:
            x0, y0, x1, y1 = self.crop
            before, after = (F.interpolate(t[:, :, y0:y1, x0:x1], size=(IMAGE_SIZE, IMAGE_SIZE),
                                           mode="bilinear", align_corners=False) for t in (before, after))
        fb = self.trunk((before - mean) / std)
        fa = self.trunk((after - mean) / std)
        return self.head(torch.cat([fb, fa, fa - fb], dim=1))

    def save(self, directory: str | Path, meta: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "checker.pt")
        (directory / "checker.json").write_text(
            json.dumps({"quarters": QUARTERS, "crop": self.crop, **meta}, indent=1))

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device = "cpu") -> "QuarterChecker":
        meta = json.loads((Path(directory) / "checker.json").read_text())
        model = cls(pretrained=False, crop=meta.get("crop"))
        model.load_state_dict(torch.load(Path(directory) / "checker.pt", map_location=device))
        return model.to(device).eval()

    @torch.no_grad()
    def added(self, before_jpeg: bytes, after_jpeg: bytes) -> dict[str, float]:
        """Probability per quarter that a piece was added between the two frames."""
        device = next(self.parameters()).device
        b = jpeg_to_tensor(before_jpeg).unsqueeze(0).to(device)
        a = jpeg_to_tensor(after_jpeg).unsqueeze(0).to(device)
        probs = torch.sigmoid(self(b, a))[0].cpu().tolist()
        return dict(zip(QUARTERS, probs))
