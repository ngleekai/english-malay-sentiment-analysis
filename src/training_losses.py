"""Class weighting that remains effective for one-sample microbatches."""
import torch
import torch.nn.functional as F


def sample_normalized_weighted_loss(logits, labels, class_weights, num_items_in_batch=None):
    """Sum weighted CE / sample count, never / sum of target weights.

    Trainer passes the total valid-label count for the accumulation window.
    Each microbatch contributes its weighted sum / that same count. During
    standalone evaluation, default to the current batch's valid-label count.
    Float32 CE keeps this calculation stable under mixed precision.
    """
    weights = class_weights.to(device=logits.device, dtype=torch.float32)
    loss_sum = F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum")
    count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    count = torch.as_tensor(count, device=logits.device, dtype=torch.float32)
    if count.numel() != 1 or count.item() <= 0:
        raise ValueError("Valid sample count must be a positive scalar")
    return loss_sum / count

