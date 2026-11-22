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
