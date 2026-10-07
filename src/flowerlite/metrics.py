
import torch
import torch.distributed as dist


class MetricsAccumulator:
    def __init__(self, num_classes: int, is_distributed: bool = False):
        self.num_classes = num_classes
        self.is_distributed = is_distributed
        
        # Accumulators
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total = 0
        self.nll_sum = 0.0
        
        # Confusion matrix
        self.conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        # ECE: 15 bins
        self.num_bins = 15
        self.bin_counts = torch.zeros(self.num_bins, dtype=torch.int64)
        self.bin_conf_sums = torch.zeros(self.num_bins, dtype=torch.float64)
        self.bin_acc_sums = torch.zeros(self.num_bins, dtype=torch.float64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain NaN or Inf.")
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError("Labels out of bounds.")
        if logits.size(0) != labels.size(0):
            raise ValueError("Length mismatch between logits and labels.")
        
        n = logits.size(0)
        if n == 0:
            return

        probs = torch.softmax(logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)
        
        self.total += n
        self.correct_top1 += (preds == labels).sum().item()
        
        if self.num_classes >= 5:
            _, top5_preds = torch.topk(logits, k=5, dim=-1)
            self.correct_top5 += (top5_preds == labels.unsqueeze(1)).sum().item()
            
        nll = torch.nn.functional.cross_entropy(logits.to(torch.float64), labels, reduction='sum')
        self.nll_sum += nll.item()
        
        for p, t in zip(preds, labels):
            self.conf_matrix[t.item(), p.item()] += 1
            
        # ECE calculation using 15 equal-width bins in [0, 1]
        bin_edges = torch.linspace(0, 1, self.num_bins + 1)
        for i in range(self.num_bins):
            lower = bin_edges[i].item()
            upper = bin_edges[i+1].item()
            if i == self.num_bins - 1:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)
            
            count = mask.sum().item()
            if count > 0:
                self.bin_counts[i] += count
                self.bin_conf_sums[i] += confidences[mask].to(torch.float64).sum().item()
                self.bin_acc_sums[i] += (preds[mask] == labels[mask]).to(torch.float64).sum().item()

    def get_state(self):
        return {
            "correct_top1": self.correct_top1,
            "correct_top5": self.correct_top5,
            "total": self.total,
            "nll_sum": self.nll_sum,
            "conf_matrix": self.conf_matrix.clone(),
            "bin_counts": self.bin_counts.clone(),
            "bin_conf_sums": self.bin_conf_sums.clone(),
            "bin_acc_sums": self.bin_acc_sums.clone(),
        }

    def load_state(self, state):
        self.correct_top1 = state["correct_top1"]
        self.correct_top5 = state["correct_top5"]
        self.total = state["total"]
        self.nll_sum = state["nll_sum"]
        self.conf_matrix = state["conf_matrix"]
        self.bin_counts = state["bin_counts"]
        self.bin_conf_sums = state["bin_conf_sums"]
        self.bin_acc_sums = state["bin_acc_sums"]

    @staticmethod
    def merge_states(states: list):
        if not states:
            raise ValueError("No states to merge")
        merged = {
            "correct_top1": sum(s["correct_top1"] for s in states),
            "correct_top5": sum(s["correct_top5"] for s in states),
            "total": sum(s["total"] for s in states),
            "nll_sum": sum(s["nll_sum"] for s in states),
            "conf_matrix": sum((s["conf_matrix"] for s in states[1:]), start=states[0]["conf_matrix"].clone()),
            "bin_counts": sum((s["bin_counts"] for s in states[1:]), start=states[0]["bin_counts"].clone()),
            "bin_conf_sums": sum((s["bin_conf_sums"] for s in states[1:]), start=states[0]["bin_conf_sums"].clone()),
            "bin_acc_sums": sum((s["bin_acc_sums"] for s in states[1:]), start=states[0]["bin_acc_sums"].clone()),
        }
        return merged

    def all_reduce(self, device="cpu"):
        if not self.is_distributed or not dist.is_initialized():
            return
            
        counts = torch.tensor([self.correct_top1, self.correct_top5, self.total], dtype=torch.int64, device=device)
        sums = torch.tensor([self.nll_sum], dtype=torch.float64, device=device)
        cm = self.conf_matrix.to(device)
        bc = self.bin_counts.to(device)
        bcs = self.bin_conf_sums.to(device)
        bas = self.bin_acc_sums.to(device)
        
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(cm, op=dist.ReduceOp.SUM)
        dist.all_reduce(bc, op=dist.ReduceOp.SUM)
        dist.all_reduce(bcs, op=dist.ReduceOp.SUM)
        dist.all_reduce(bas, op=dist.ReduceOp.SUM)
        
        self.correct_top1 = counts[0].item()
        self.correct_top5 = counts[1].item()
        self.total = counts[2].item()
        self.nll_sum = sums[0].item()
        self.conf_matrix = cm.cpu()
        self.bin_counts = bc.cpu()
        self.bin_conf_sums = bcs.cpu()
        self.bin_acc_sums = bas.cpu()

    def compute(self):
        if self.total == 0:
            return {
                "top1": 0.0,
                "top5": 0.0,
                "macro_f1": 0.0,
                "nll": 0.0,
                "ece": 0.0,
                "confusion_matrix": self.conf_matrix.tolist(),
                "total": 0
            }
        
        top1 = (self.correct_top1 / self.total) * 100.0
        top5 = (self.correct_top5 / self.total) * 100.0 if self.num_classes >= 5 else 0.0
        nll = self.nll_sum / self.total
        
        # Macro F1
        f1_sum = 0.0
        for c in range(self.num_classes):
            tp = self.conf_matrix[c, c].item()
            fp = self.conf_matrix[:, c].sum().item() - tp
            fn = self.conf_matrix[c, :].sum().item() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall > 0:
                f1_sum += 2 * (precision * recall) / (precision + recall)
                
        macro_f1 = f1_sum / self.num_classes
        
        # ECE
        ece = 0.0
        for i in range(self.num_bins):
            if self.bin_counts[i] > 0:
                acc = self.bin_acc_sums[i] / self.bin_counts[i]
                conf = self.bin_conf_sums[i] / self.bin_counts[i]
                weight = self.bin_counts[i] / self.total
                ece += weight * abs(acc - conf)
                
        return {
            "top1": top1,
            "top5": top5,
            "macro_f1": macro_f1,
            "nll": nll,
            "ece": ece,
            "confusion_matrix": self.conf_matrix.tolist(),
            "total": self.total
        }
