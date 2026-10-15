"""Length-aware CTC loss and detached epoch statistics.

This framework adaptation is an unexecuted draft awaiting user review.
"""

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["CTC"]


def minimum_ctc_frames(target):
    """Each adjacent repeated label needs an intervening blank frame."""
    repeats = sum(left == right for left, right in zip(target, target[1:]))
    return len(target) + repeats


class CTC(nn.Module):
    """Return a differentiable CTC loss; accumulate only detached statistics.

    forward accepts logits [batch, frames, classes], output_lengths [batch],
    flattened character targets, and target_lengths [batch]. The vocabulary
    includes blank, but target transcripts must exclude blank.

    MPS logits use CPU CTC with an autograd-preserving device transfer.
    This framework adaptation has not been executed or tested.
    """

    def __init__(self, blank=0):
        super().__init__()
        if blank < 0:
            raise ValueError("blank must be a nonnegative character index")
        self.blank = blank
        self.reset()

    def forward(self, logits, output_lengths, targets_flat, target_lengths,
                utterance_ids=None):
        if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("Expected nonempty logits [batch, frames, classes]")
        batch_size, frames, classes = logits.shape
        if self.blank >= classes:
            raise ValueError("blank must be inside the model's character vocabulary")
        if (output_lengths.ndim != 1 or target_lengths.ndim != 1
                or len(output_lengths) != batch_size
                or len(target_lengths) != batch_size
                or targets_flat.ndim != 1):
            raise ValueError("Expected flattened targets and one pair of lengths per utterance")
        if utterance_ids is not None and len(utterance_ids) != batch_size:
            raise ValueError("Expected one utterance ID per batch item")

        input_lengths_cpu = output_lengths.detach().cpu().long()
        target_lengths_cpu = target_lengths.detach().cpu().long()
        targets_cpu = targets_flat.detach().cpu().long()
        if (input_lengths_cpu <= 0).any() or (input_lengths_cpu > frames).any():
            raise ValueError("Invalid output lengths")
        if (int(target_lengths_cpu.sum()) != targets_cpu.numel()
                or (target_lengths_cpu <= 0).any()):
            raise ValueError("Invalid flattened targets or target lengths")
        if (targets_cpu == self.blank).any():
            raise ValueError("Training targets must not contain CTC blanks")
        if (targets_cpu < 0).any() or (targets_cpu >= classes).any():
            raise ValueError("Target character index is outside the vocabulary")

        offset = 0
        for index, length in enumerate(target_lengths_cpu.tolist()):
            target = targets_cpu[offset:offset + length].tolist()
            offset += length
            required = minimum_ctc_frames(target)
            available = int(input_lengths_cpu[index])
            if required > available:
                label = utterance_ids[index] if utterance_ids is not None else index
                raise ValueError(
                    f"Impossible CTC alignment for {label}: needs {required} frames, "
                    f"has {available}"
                )

        log_probs = logits.float().log_softmax(-1).transpose(0, 1)
        # Copying to CPU remains differentiable; do not detach log_probs or loss.
        loss_device = torch.device("cpu") if logits.device.type == "mps" else logits.device
        loss = F.ctc_loss(
            log_probs.to(loss_device), targets_cpu.to(loss_device),
            input_lengths_cpu, target_lengths_cpu, blank=self.blank,
            reduction="mean", zero_infinity=False,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CTC loss")

        # PyTorch's mean averages target-length-normalized losses over utterances.
        self.weighted_total_loss += loss.detach().cpu() * batch_size
        self.num_samples += batch_size
        return loss

    def compute(self):
        """Detached sample-weighted mean over the current epoch's batches."""
        if self.num_samples == 0:
            raise RuntimeError("No CTC losses accumulated in this epoch")
        return self.weighted_total_loss / self.num_samples

    def reset(self):
        self.weighted_total_loss = 0.0
        self.num_samples = 0
