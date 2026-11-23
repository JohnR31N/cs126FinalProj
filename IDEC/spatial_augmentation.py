"""Whole-image spatial transforms that keep the handwriting and mark inside the crop."""
import math

import torch
from torch.nn import functional as F


class Translation:
    def __init__(self, images):
        self.images = images
        pixels = images.view(-1, 28, 28) > 0
        axes = torch.arange(28, device=images.device)
        bounds = []
        for occupied in (pixels.any(2), pixels.any(1)):
            first = torch.where(occupied, axes, 28).amin(1)
            last = torch.where(occupied, axes, -1).amax(1)
            bounds.append(((-first).clamp(min=-3), (27-last).clamp(max=3)))
        self.bounds = bounds
        self.rows, self.cols = torch.meshgrid(axes, axes, indexing="ij")
        counts = (bounds[0][1]-bounds[0][0]+1) * (bounds[1][1]-bounds[1][0]+1)
        self.audit = {"mean_legal_offsets": counts.float().mean().item(),
                      "min_legal_offsets": counts.min().item(), "max_legal_offsets": counts.max().item(),
                      "mean_identity_probability": counts.float().reciprocal().mean().item()}

    def __call__(self, generator):
        offsets = []
        for low, high in self.bounds:
            random = torch.rand(len(low), generator=generator, device=low.device)
            offsets.append(low + (random * (high - low + 1)).long())
        source_rows = (self.rows - offsets[0][:, None, None]) % 28
        source_cols = (self.cols - offsets[1][:, None, None]) % 28
        return self.images.gather(1, (28 * source_rows + source_cols).flatten(1))


class Rotation:
    angles = (-12, -8, -4, 0, 4, 8, 12)

    def __init__(self, images, full_scale):
        self.views = torch.empty((len(self.angles), *images.shape), device=images.device, dtype=images.dtype)
        self.legal = torch.empty((len(images), len(self.angles)), device=images.device, dtype=torch.bool)
        self.index = torch.arange(len(images), device=images.device)
        for k, angle in enumerate(self.angles):
            if angle == 0:
                self.views[k] = images
                self.legal[:, k] = True
                continue
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            matrix = images.new_tensor([[c, -s, 0], [s, c, 0]])
            for ids in self.index.split(512):
                padded = F.pad(images[ids].reshape(-1, 1, 28, 28), (5, 5, 5, 5))
                grid = F.affine_grid(matrix.expand(len(ids), -1, -1), padded.shape, align_corners=False)
                rotated = F.grid_sample(padded, grid, mode="bilinear", align_corners=False)
                crop = rotated[:, :, 5:-5, 5:-5]
                self.views[k, ids] = crop.reshape(-1, 784)
                crop.zero_()
                self.legal[ids, k] = rotated.flatten(1).amax(1) <= full_scale * 1e-6
        counts = self.legal.sum(1)
        self.audit = {"angles_degrees": list(self.angles), "legal_image_counts": self.legal.sum(0).tolist(),
                      "identity_only_images": int((counts == 1).sum().item()),
                      "mean_identity_probability": counts.float().reciprocal().mean().item()}

    def __call__(self, generator):
        scores = torch.rand(self.legal.shape, device=self.legal.device, generator=generator)
        selected = scores.masked_fill(~self.legal, -1).argmax(1)
        return self.views[selected, self.index]
