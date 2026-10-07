import torch
from torch import nn

from train import EMA


def test_ema_warmup():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(1.0))
            
    model = DummyModel()
    ema = EMA(model, decay=0.9998)
    
    # Check updates
    assert ema.updates == 0
    
    # Step 1
    model.p.data = torch.tensor(2.0)
    ema.update()
    assert ema.updates == 1
    # decay_t = min(0.9998, 2.0 / 11.0) = 0.181818
    expected_d = 2.0 / 11.0
    expected_val = expected_d * 1.0 + (1.0 - expected_d) * 2.0
    assert torch.allclose(ema.shadow["p"], torch.tensor(expected_val))
    
    # Step 10
    for _ in range(9):
        model.p.data = torch.tensor(3.0)
        ema.update()
        
    assert ema.updates == 10
    # decay_t for step 10 = 11.0 / 20.0 = 0.55
    # decay is growing up to 0.9998.
