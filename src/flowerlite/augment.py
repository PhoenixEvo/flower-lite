import math

import torch
import torchvision.transforms.v2 as transforms


def get_train_transform(mean, std, trivial_augment=True):
    ops = [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if trivial_augment:
        ops.append(transforms.TrivialAugmentWide())
    
    ops.extend([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms.Compose(ops)

def get_eval_transform(mean, std):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std)
    ])

def mixup_batch(x, y, num_classes, alpha, device):
    batch_size = x.size(0)
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    index = torch.randperm(batch_size, device=device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes).float()
    
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y

def cutmix_batch(x, y, num_classes, alpha, device):
    batch_size, _, h, w = x.size()
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    index = torch.randperm(batch_size, device=device)

    cut_rat = math.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)

    realized_area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1. - realized_area / (h * w)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes).float()

    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y

def apply_mix(x, y, num_classes, mixup_prob, cutmix_prob, alpha, device):
    p = torch.rand(1).item()
    if p < mixup_prob:
        return mixup_batch(x, y, num_classes, alpha, device), "mixup"
    elif p < mixup_prob + cutmix_prob:
        return cutmix_batch(x, y, num_classes, alpha, device), "cutmix"
    else:
        y_soft = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        return (x, y_soft), "none"
