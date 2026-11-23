import numpy as np
import torch

from varied_backgrounds import template_bank, render
from varied_study import finetune
from idec_final.model import Autoencoder
from idec_final.experiment import finetune as original_finetune
from idec_final.utils import seed_all


def test_matched_energy_symbols_preserve_faint_ink():
    bank, fixed, attr = template_bank()
    assert attr.shape == (96, 4) and len(np.unique(bank.reshape(96, -1), axis=0)) == 96
    clean = np.zeros((96, 28, 28), dtype=np.float32)
    clean[:, 13, 13] = 1/255
    boxes = np.broadcast_to([[[0, 0], [23, 23]]], (96, 2, 2))
    types = np.repeat(np.arange(96)[:, None], 2, axis=1)
    observed = render(clean.reshape(96,-1), boxes, types, bank).reshape(96,28,28)
    matched = render(clean.reshape(96,-1), boxes, types, fixed).reshape(96,28,28)
    np.testing.assert_array_equal(observed[:, 6:22, 6:22], clean[:, 6:22, 6:22])
    np.testing.assert_allclose(((observed-clean)**2).sum((1,2)), ((matched-clean)**2).sum((1,2)), rtol=4e-7)


def test_reconstruction_target_extension_retains_standard_updates():
    import copy
    seed_all(3)
    model = Autoencoder([8, 5, 2])
    reference = copy.deepcopy(model)
    images = torch.rand(18, 8)
    centers = torch.rand(3, 2)
    cfg = dict(lr=.001, gamma=.1, batch_size=6, target_interval=1, updates=3)
    a, trace = finetune(model, centers, images, images, cfg)
    b, other = original_finetune(reference, centers, images, None, "idec", cfg)
    assert torch.equal(a, b) and trace == other
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters()))
