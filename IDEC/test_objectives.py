"""Independent numerical checks for the baseline and the intervention."""
import numpy as np
import torch

from idec_final.metrics import evaluate
from idec_final.model import student_q, target_distribution
from idec_final.stripes import observe, orientations, projected_mse, templates


def test_assignments_and_target_match_hand_calculation():
    z=torch.tensor([[0.],[1.],[3.]],dtype=torch.float64)
    centers=torch.tensor([[0.],[2.]],dtype=torch.float64)
    q=student_q(z,centers)
    expected=np.array([[5/6,1/6],[1/2,1/2],[1/6,5/6]])
    np.testing.assert_allclose(q,expected)
    # Unequal cluster frequencies expose omission of the frequency correction.
    q=torch.cat([q,q[:1]])
    weighted=q.numpy()**2 / np.sum(q.numpy(),axis=0)
    np.testing.assert_allclose(target_distribution(q),weighted/weighted.sum(axis=1,keepdims=True))


def test_projection_value_and_gradient_agree_with_dense_oracle():
    d=torch.tensor([.6,.8,0.],dtype=torch.float64)
    residual=torch.tensor([[1.,-2.,3.],[2.,0.,-1.]],dtype=torch.float64,requires_grad=True)
    projector=np.eye(3)-np.outer(d,d)
    oracle=residual.detach().numpy() @ projector
    loss=projected_mse(residual,d)
    np.testing.assert_allclose(loss.detach(),np.mean(oracle**2))
    loss.backward()
    np.testing.assert_allclose(residual.grad,2*oracle/residual.numel(),atol=1e-12)


def test_counterfactual_changes_only_the_declared_direction():
    patterns,d=templates("cpu")
    clean=torch.linspace(0,1,784).expand(6,-1)
    orientation=torch.from_numpy(orientations(6,1701))
    assert orientation.sum()==3
    a=observe(clean,orientation,.8,patterns)
    b=observe(clean,1-orientation,.8,patterns)
    assert (a-b).abs().max()>1
    assert projected_mse(a-b,d)<1e-11
    torch.testing.assert_close(observe(clean,orientation,0,patterns), clean*(255/50))


def test_test_labels_cannot_reassign_cluster_names():
    yy=np.repeat(np.arange(10),2)
    train=(yy+1)%10
    # If test labels were allowed to rematch cluster IDs, this score would be one.
    r=evaluate(yy,train,yy,(train+1)%10,train,yy%2)
    assert r["train_acc"]==1 and r["digit_acc"]==0 and r["flip_rate"]==1


def test_stage_pretraining_matches_the_original_recipe_except_for_the_projected_error():
    from idec_final.experiment import pretrain
    from train_stages import pretrain_projected

    samples = torch.randn(16, 4, generator=torch.Generator().manual_seed(126))
    samples[:, 0] += 1
    cfg = dict(dims=[4, 5, 2], lr=0.01, pretrain_epochs=3, batch_size=3, clusters=2, kmeans_restarts=2)
    baseline, centers = pretrain(samples, cfg, seed=3, device=samples.device)
    # A zero vector makes the residual transformation the identity.
    replay, replay_centers = pretrain_projected(samples, torch.zeros(4), cfg, seed=3)
    for left, right in zip(baseline.parameters(), replay.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(centers, replay_centers, rtol=0, atol=0)
    assert torch.unique(centers, dim=0).size(0) == cfg["clusters"]

    # Removing the first pixel error gives its output bias exactly zero gradient.
    projected, _ = pretrain_projected(samples, torch.tensor([1.0, 0.0, 0.0, 0.0]), cfg, seed=3)
    assert projected.decoder[-1].bias[0].item() == 0
    assert abs(baseline.decoder[-1].bias[0].item()) > 0.001


def test_stage_digest_accepts_only_line_endings_in_text_files(tmp_path):
    import hashlib
    from analyze_stages import matches_training_digest

    recorded = hashlib.sha256(b"x = 1\r\n").hexdigest()
    source = tmp_path / "model.py"
    source.write_bytes(b"x = 1\n")
    assert matches_training_digest(source, recorded)
    source.write_bytes(b"x = 2\n")
    assert not matches_training_digest(source, recorded)
    binary = tmp_path / "evaluation.npz"
    binary.write_bytes(b"x = 1\n")
    assert not matches_training_digest(binary, recorded)
