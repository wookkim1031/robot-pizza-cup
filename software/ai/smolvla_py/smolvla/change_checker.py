"""Change checker: was a topping ADDED to a pizza sector between two scene-camera frames?

Runtime port of the DevLab checker (gpu2 ~/v2/scripts/change_checker.py + h_train_change.py;
docs in ~/v2/scripts/CHANGE_CHECKER.md). Same model and preprocessing, but the ResNet18 is
built without ImageNet weights (the checkpoint overwrites them), so loading needs no network.

Reliable for pepperoni (99.6% detected, 0-1 false "added" in 1206 held-out pairs at 0.5-0.7);
NOT for cheese (51% detected). Sectors TL TR BR BL CENTER use the manifest names; GAP has no box.

Frames: RGB uint8 (224, 224, 3) from the scene camera, arm at home in both, same camera pose
and pizza spot as the training data (pizza circle (118, 131) r 28).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_model() -> nn.Module:
    """ResNet18 with a 6-channel stem ([after, after - before]) and 2 logits (cheese, pepperoni)."""
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(512, 2)
    return m


class ChangeChecker:
    def __init__(self, path: Path | str, device: str | torch.device | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(Path(path).expanduser(), map_location=self.dev)
        self.roi: dict[str, list[int]] = ck["roi"]
        self.labels: list[str] = ck["labels"]
        self.size, self.crop = ck["size"], ck["crop"]
        self.diff_scale = ck.get("diff_scale", 0.25)
        self.image_size = tuple(ck.get("image_size", (224, 224)))
        self.model = build_model()
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(self.dev).eval()

    def _crop(self, img: np.ndarray, sector: str) -> torch.Tensor:
        im = Image.fromarray(np.asarray(img))
        if im.size != self.image_size:
            raise ValueError(f"frame is {im.size}, the checker's sector boxes are for {self.image_size}")
        im = im.crop(self.roi[sector]).resize((self.size, self.size), Image.BICUBIC)
        return TF.center_crop(TF.to_tensor(im), [self.crop, self.crop])

    def _input(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        return torch.cat([(after - MEAN) / STD, (after - before) / self.diff_scale], 0)

    @torch.no_grad()
    def scan(self, before: np.ndarray, after: np.ndarray) -> dict[str, dict[str, float]]:
        """{sector: {"cheese": p, "pepperoni": p}} for every sector, in one batch."""
        sectors = list(self.roi)
        x = torch.stack([self._input(self._crop(before, s), self._crop(after, s)) for s in sectors]).to(self.dev)
        p = torch.sigmoid(self.model(x)).cpu().tolist()
        return {s: dict(zip(self.labels, ps)) for s, ps in zip(sectors, p)}

    def check_step(self, before: np.ndarray, after: np.ndarray, target: str, topping: str,
                   thr: float = 0.65) -> tuple[bool, float, dict[str, float]]:
        """(added to target?, p for target, {other sector: p} where the topping also appeared)."""
        res = self.scan(before, after)
        p = res[target][topping]
        elsewhere = {s: round(v[topping], 3) for s, v in res.items() if s != target and v[topping] > thr}
        return p > thr, p, elsewhere
