
import pytest
import torch

from flowerlite.metrics import MetricsAccumulator


def test_metrics_hand_computed():
    from sklearn.metrics import f1_score, log_loss
    
    # 5 classes
    logits = torch.tensor([
        [2.0, -1.0, 0.5, 0.0, 1.0],  # class 0 pred
        [-1.0, 3.0, 0.5, 0.0, 0.0],  # class 1 pred
        [0.0,  0.0, 0.0, 0.0, 2.0],  # class 4 pred
        [-2.0, -2.0, 3.0, 0.0, 0.0], # class 2 pred
        [0.0,  0.0, 0.0, 4.0, 0.0]   # class 3 pred
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 3, 2, 3], dtype=torch.int64) # Class 3 is wrong, it pred 4
    
    acc = MetricsAccumulator(num_classes=5)
    acc.update(logits, labels)
    res = acc.compute()
    
    assert res["top1"] == 80.0
    # Top 5 should be 100% since there are 5 classes
    assert res["top5"] == 100.0
    
    preds = [0, 1, 4, 2, 3]
    y_true = [0, 1, 3, 2, 3]
    
    sk_f1 = f1_score(y_true, preds, average='macro', labels=[0,1,2,3,4])
    assert abs(res["macro_f1"] - sk_f1) < 1e-10
    
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    sk_nll = log_loss(y_true, probs.numpy(), labels=[0,1,2,3,4])
    assert abs(res["nll"] - sk_nll) < 1e-10

def test_confusion_matrix():
    logits = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.int64)
    acc = MetricsAccumulator(num_classes=2)
    acc.update(logits, labels)
    res = acc.compute()
    cm = res["confusion_matrix"]
    assert sum(sum(row) for row in cm) == 2

def test_ece_perfect():
    logits = torch.tensor([
        [0.0, 100.0], # prob approx [0, 1], conf 1.0, acc 1.0
        [0.0, 0.0],   # prob [0.5, 0.5], conf 0.5, acc 0.5
    ], dtype=torch.float32)
    # If prob=0.5 and pred=0 or 1, let's say pred is 0 and label is 1 (wrong)
    # For perfect ECE, confidence must equal accuracy in each bin.
    # We will hand craft probabilities.
    probs = torch.tensor([
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
    ], dtype=torch.float32)
    logits = torch.log(probs)
    # Confidence is 0.9. Accuracy must be 0.9. 
    # 9 correct, 1 incorrect.
    labels = torch.tensor([1,1,1,1,1,1,1,1,1,0], dtype=torch.int64)
    
    acc = MetricsAccumulator(num_classes=2)
    acc.update(logits, labels)
    res = acc.compute()
    assert abs(res["ece"]) < 1e-6

def test_invalid_inputs():
    acc = MetricsAccumulator(num_classes=2)
    with pytest.raises(ValueError):
        acc.update(torch.tensor([[float('nan'), 0.0]]), torch.tensor([0]))
    with pytest.raises(ValueError):
        acc.update(torch.tensor([[0.0, 0.0]]), torch.tensor([2]))
    with pytest.raises(ValueError):
        acc.update(torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([0]))

def _run_distributed(rank, world_size, sync_file):

    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file:///{sync_file}", rank=rank, world_size=world_size)
    
    acc = MetricsAccumulator(num_classes=3, is_distributed=True)
    if rank == 0:
        logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        acc.update(logits, labels)
    else:
        # Empty shard
        pass
        
    acc.all_reduce()
    res = acc.compute()
    dist.destroy_process_group()
    return res

def test_all_reduce_correctness(tmp_path):

    import torch.multiprocessing as mp
    sync_file = str(tmp_path / "sync_file").replace('\\', '/')
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(2)
    try:
        results = pool.starmap(_run_distributed, [(0, 2, sync_file), (1, 2, sync_file)])
    except RuntimeError as e:
        if "unsupported gloo device" in str(e) or "client socket has failed" in str(e):
            pytest.skip(f"REQUIRED-ON-KAGGLE: {e!s}")
        else:
            raise
    pool.close()
    pool.join()
    
    r0 = results[0]
    r1 = results[1]
    
    # Expected: 2 total, 100% top1
    assert r0["top1"] == 100.0
    assert r1["top1"] == 100.0
    assert r0["total"] == 2

def test_merge_states():
    # Merge 3 states, including an EMPTY state.
    acc1 = MetricsAccumulator(num_classes=3)
    acc2 = MetricsAccumulator(num_classes=3)
    acc_empty = MetricsAccumulator(num_classes=3)
    
    logits1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    labels1 = torch.tensor([0, 1], dtype=torch.int64)
    acc1.update(logits1, labels1)
    
    logits2 = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    labels2 = torch.tensor([2], dtype=torch.int64)
    acc2.update(logits2, labels2)
    
    merged_state = MetricsAccumulator.merge_states([
        acc1.get_state(),
        acc_empty.get_state(),
        acc2.get_state()
    ])
    
    merged_acc = MetricsAccumulator(num_classes=3)
    merged_acc.load_state(merged_state)
    merged_res = merged_acc.compute()
    
    # Compare with single process
    acc_single = MetricsAccumulator(num_classes=3)
    acc_single.update(torch.cat([logits1, logits2]), torch.cat([labels1, labels2]))
    single_res = acc_single.compute()
    
    assert abs(merged_res["top1"] - single_res["top1"]) < 1e-10
    assert abs(merged_res["macro_f1"] - single_res["macro_f1"]) < 1e-10
    assert abs(merged_res["nll"] - single_res["nll"]) < 1e-10
    assert abs(merged_res["ece"] - single_res["ece"]) < 1e-10
    assert merged_res["confusion_matrix"] == single_res["confusion_matrix"]
    assert merged_res["total"] == 3

def test_ece_second_hand_computed():
    # ECE bins are [0, 1/15), ..., [14/15, 1.0]. (left-closed, right-open except last which is right-closed).
    # Actually my implementation does: i == 14 -> [14/15, 1.0], else [i/15, (i+1)/15).
    # Let's verify this.
    probs = torch.tensor([
        [0.2, 0.8], # conf 0.8, bin 12 ([0.8, 0.866))
        [0.1, 0.9], # conf 0.9, bin 13 ([0.866, 0.933))
        [0.4, 0.6]  # conf 0.6, bin 9 ([0.6, 0.666))
    ], dtype=torch.float32)
    logits = torch.log(probs)
    labels = torch.tensor([1, 0, 1], dtype=torch.int64) # acc: 1, 0, 1
    
    # Expected ECE:
    # total = 3
    # bin 12 (0.8): count 1, acc 1.0, conf 0.8. error: |1.0 - 0.8| = 0.2, weight 1/3
    # bin 13 (0.9): count 1, acc 0.0, conf 0.9. error: |0.0 - 0.9| = 0.9, weight 1/3
    # bin 9 (0.6): count 1, acc 1.0, conf 0.6. error: |1.0 - 0.6| = 0.4, weight 1/3
    # ECE = (0.2 + 0.9 + 0.4) / 3 = 1.5 / 3 = 0.5
    
    acc = MetricsAccumulator(num_classes=2)
    acc.update(logits, labels)
    res = acc.compute()
    assert abs(res["ece"] - 0.5) < 1e-6

def test_top5_hand_computed():
    # 6 classes
    logits = torch.tensor([
        [0, 1, 2, 3, 4, 5], # top 5: 1,2,3,4,5. Label 0 is NOT in top 5.
        [0, 1, 2, 3, 4, 5], # top 5: 1,2,3,4,5. Label 1 IS in top 5.
        [5, 4, 3, 2, 1, 0], # top 5: 0,1,2,3,4. Label 5 is NOT in top 5.
        [5, 4, 3, 2, 1, 0]  # top 5: 0,1,2,3,4. Label 4 IS in top 5.
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 5, 4], dtype=torch.int64)
    # Expected top5 acc: 2 / 4 = 50.0%
    acc = MetricsAccumulator(num_classes=6)
    acc.update(logits, labels)
    res = acc.compute()
    assert res["top5"] == 50.0

def test_merge_unequal_shards_never_average():
    # 7 and 3 samples. If we average per-rank metrics, top1 would be (top1_shard1 + top1_shard2)/2
    acc1 = MetricsAccumulator(num_classes=2)
    acc2 = MetricsAccumulator(num_classes=2)
    
    # 7 correct samples in acc1
    logits1 = torch.tensor([[0.0, 1.0]] * 7, dtype=torch.float32)
    labels1 = torch.tensor([1] * 7, dtype=torch.int64)
    acc1.update(logits1, labels1)
    # top1 of acc1 = 100%
    
    # 3 incorrect samples in acc2
    logits2 = torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float32)
    labels2 = torch.tensor([0] * 3, dtype=torch.int64)
    acc2.update(logits2, labels2)
    # top1 of acc2 = 0%
    
    merged_state = MetricsAccumulator.merge_states([acc1.get_state(), acc2.get_state()])
    merged_acc = MetricsAccumulator(num_classes=2)
    merged_acc.load_state(merged_state)
    res = merged_acc.compute()
    
    # total samples = 10. correct = 7. Expected top1 = 70.0%
    # If it averaged 100% and 0%, it would be 50.0%.
    assert res["top1"] == 70.0
    assert res["total"] == 10
