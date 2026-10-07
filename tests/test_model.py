import torch
from fvcore.nn import FlopCountAnalysis

from flowerlite.models.flowerlite import FlowerLiteL


def test_model_shapes_finite_backward():
    model = FlowerLiteL(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
    
    loss = out.sum()
    loss.backward()
    
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

def test_params_flops_gate():
    model_10 = FlowerLiteL(num_classes=10)
    model_100 = FlowerLiteL(num_classes=100)
    
    params_10 = sum(p.numel() for p in model_10.parameters() if p.requires_grad)
    params_100 = sum(p.numel() for p in model_100.parameters() if p.requires_grad)
    
    x = torch.randn(1, 3, 32, 32)
    flops_10 = FlopCountAnalysis(model_10, x).total()
    flops_100 = FlopCountAnalysis(model_100, x).total()
    
    # 1 MAC = 2 FLOPs, so we check MACs or FLOPs? The gate says FLOPs < 1G (which is 0.5G MACs).
    # FlopCountAnalysis counts MACs! So total() gives MACs.
    assert params_10 < 6_000_000
    assert params_100 < 6_000_000
    assert flops_10 * 2 < 1_000_000_000
    assert flops_100 * 2 < 1_000_000_000
    
    print(f"\nK=10: Params={params_10}, MACs={flops_10}, FLOPs={flops_10*2}")
    print(f"K=100: Params={params_100}, MACs={flops_100}, FLOPs={flops_100*2}")

def test_tiny_overfit():
    model = FlowerLiteL(num_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    
    for _ in range(50):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    out = model(x)
    pred = out.argmax(dim=1)
    acc = (pred == y).float().mean()
    assert acc > 0.8 # should overfit well
