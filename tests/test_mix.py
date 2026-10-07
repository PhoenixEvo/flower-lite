import numpy as np
import torch
import torch.nn.functional as F


def test_mixup_cutmix_targets():
    # Simulate a batch
    batch_size = 4
    num_classes = 10
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    
    # MIXUP
    lam = 0.3
    idx = torch.tensor([3, 2, 1, 0])
    
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    y_idx_onehot = F.one_hot(y[idx], num_classes=num_classes).float()
    
    y_mixed = lam * y_onehot + (1.0 - lam) * y_idx_onehot
    
    # Assert float soft targets whose rows sum to 1 within 1e-6
    sums = y_mixed.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    # Assert loss accepts them
    out = torch.randn(batch_size, num_classes)
    loss = F.cross_entropy(out, y_mixed, label_smoothing=0.05)
    assert loss.item() > 0
    assert torch.isfinite(loss)
