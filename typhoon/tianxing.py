"""Thin adapter preserving the supplied TianXing model's native extra features."""

import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
import xarray as xr


class TianXing:
    def __init__(self, model_root, device="cuda:0", extra_features=True):
        # Full FP32 is needed for reliable small initial-state derivatives.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.root = Path(model_root)
        sys.path.insert(0, str(self.root))
        from weaformer2.backbone import Weaformer_model

        self.device = torch.device(device)
        self.model = Weaformer_model()
        weights = self.root / "assets/model_extra.pth"
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.as_tensor(
            np.load(self.root / "assets/global_means.npy")[:, :73], device=self.device
        )
        self.std = torch.as_tensor(
            np.load(self.root / "assets/global_stds.npy")[:, :73], device=self.device
        )
        self.extra_features = extra_features
        if extra_features:
            with xr.open_dataset(self.root / "assets/land_constant.nc") as ds:
                lsm = np.asarray(ds.lsm).reshape(721, 1440)
                z = np.maximum(np.asarray(ds.z).reshape(721, 1440), 0)
            lat = np.broadcast_to(
                np.sin(np.deg2rad(np.linspace(90, -90, 721)))[:, None], (721, 1440)
            )
            self.constants = np.stack([lat, lsm, z / z.max()]).astype(np.float32)
        self.provenance = {
            "weights": weights.name,
            "weights_sha256": self.digest(weights),
            "extra_features": extra_features,
            "torch_version": torch.__version__,
            "precision": "float32; TF32 disabled",
            "source_sha256": {
                p.name: self.digest(p)
                for p in (self.root / "weaformer2").glob("*.py")
                if " " not in p.name
            },
        }

    @staticmethod
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def normalize(self, x):
        return (torch.as_tensor(x, device=self.device) - self.mean) / self.std

    def physical(self, x):
        return x * self.std + self.mean

    def __call__(self, x, valid_time):
        if not self.extra_features:
            return self.model(x)
        hour = valid_time.hour + valid_time.minute / 60
        solar = np.broadcast_to(
            np.cos(np.deg2rad(np.arange(1440) * 0.25) + hour / 24 * 2 * np.pi)[None, :],
            (721, 1440),
        ).astype(np.float32)
        # Official run concatenates geographic/land constants then the time map.
        extra = torch.as_tensor(
            np.concatenate([self.constants, solar[None]], axis=0)[None],
            device=self.device,
        )
        return self.model(x, extra)
