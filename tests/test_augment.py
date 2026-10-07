import torch

from flowerlite.augment import cutmix_batch, get_eval_transform, mixup_batch


def test_eval_transform_deterministic():
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    t = get_eval_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    out1 = t(img)
    for _ in range(9):
        assert torch.allclose(out1, t(img))

def test_mixup_row_sums():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    mx, my = mixup_batch(x, y, 10, alpha=1.0, device="cpu")
    sums = my.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_cutmix_row_sums_and_lambda():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    # Force lambda to be predictable or check lambda logic
    mx, my = cutmix_batch(x, y, 10, alpha=1.0, device="cpu")
    sums = my.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    # Verify lambda matching area
    # In cutmix_batch we compute lam = 1. - realized_area / 1024
    # The max value in my is max(lam, 1-lam) for some rows if y are disjoint
    # We just ensure it's mathematically consistent
    max_val = my[0].max().item()
    lam = max_val if max_val > 0.5 else 1 - max_val
    if max_val != 1.0:
        area = round((1 - lam) * 1024)
        assert abs(1 - area/1024 - lam) < 1e-6

def test_apply_mix_probabilities():
    from flowerlite.augment import apply_mix
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    
    counts = {"mixup": 0, "cutmix": 0, "none": 0}
    for _ in range(1000):
        # mixup 30%, cutmix 40%, none 30%
        _, applied = apply_mix(x, y, 10, mixup_prob=0.3, cutmix_prob=0.4, alpha=1.0, device="cpu")
        counts[applied] += 1
        
    # Tolerance of ±0.05 (50 counts out of 1000)
    assert 250 <= counts["mixup"] <= 350
    assert 350 <= counts["cutmix"] <= 450
    assert 250 <= counts["none"] <= 350
