"""Transformer–CTC acoustic model adapted into the SLM framework.

This framework adaptation is an unexecuted draft awaiting user review.
"""

import math

import torch
from torch import nn


__all__ = ["Transformer_STT"]


def padding_mask(lengths, max_length):
    """Return True for positions outside each utterance's valid frames."""
    positions = torch.arange(max_length, device=lengths.device)
    return positions[None, :] >= lengths[:, None]


class Transformer_STT(nn.Module):
    """Log-Mel frames → convolutional subsampling → offline Transformer.

    Inputs are features [batch, frames, n_mels] and feature_lengths [batch].
    Outputs are logits [batch, subsampled_frames, vocab_size] and the valid
    output lengths. Vocabulary size includes the CTC blank. This model does
    not tokenize transcripts or perform CTC decoding.

    This framework adaptation has not been executed or tested.
    """

    def __init__(self, vocab_size=29, n_mels=80, channels=(32, 64),
                 d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
                 dropout=0.1, max_positions=8192):
        super().__init__()
        if min(vocab_size, n_mels, d_model, nhead, num_layers,
               dim_feedforward, max_positions) <= 0:
            raise ValueError("Model dimensions and layer counts must be positive")
        if d_model % nhead or d_model % 2 or len(channels) != 2:
            raise ValueError("Expected even d_model divisible by nhead and two convolution widths")
        if any(width <= 0 for width in channels):
            raise ValueError("Convolution widths must be positive")

        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels[0], 3, stride=2, padding=1),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
        ])
        mel_out = (n_mels + 1) // 2
        mel_out = (mel_out + 1) // 2
        self.projection = nn.Linear(channels[-1] * mel_out, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Construct layers independently instead of cloning one initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

        positions = torch.arange(max_positions).float().unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_positions, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positions", encoding, persistent=False)

    @staticmethod
    def output_lengths(lengths):
        """Lengths after two kernel-3, stride-2, padding-1 convolutions."""
        return ((lengths + 1) // 2 + 1) // 2

    def forward(self, features, feature_lengths):
        if (features.ndim != 3 or feature_lengths.ndim != 1
                or len(feature_lengths) != len(features)):
            raise ValueError("Expected [batch, frames, mel] and [batch] lengths")
        if features.shape[0] == 0 or features.shape[2] != self.n_mels:
            raise ValueError("Expected a nonempty batch with the configured n_mels")
        lengths = feature_lengths.to(device=features.device, dtype=torch.long)
        if (lengths <= 0).any() or (lengths > features.shape[1]).any():
            raise ValueError("Invalid feature lengths")

        mask = padding_mask(lengths, features.shape[1])
        x = features.masked_fill(mask[:, :, None], 0)
        x = x.transpose(1, 2).unsqueeze(1)  # [batch, 1, mel, frames]
        for conv in self.convs:
            x = torch.nn.functional.gelu(conv(x))
            lengths = (lengths + 1) // 2
            # Remove padded activations before the next convolution can use them.
            mask = padding_mask(lengths, x.shape[-1])
            x = x.masked_fill(mask[:, None, None, :], 0)

        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.input_norm(self.projection(x))
        if x.shape[1] > len(self.positions):
            raise ValueError("Sequence exceeds max_positions; adjust config instead of truncating")
        mask = padding_mask(lengths, x.shape[1])
        x = (x + self.positions[:x.shape[1]]).masked_fill(mask[:, :, None], 0)
        for layer in self.layers:
            # Padding is masked; no causal mask, so all valid audio is available.
            x = layer(x, src_key_padding_mask=mask)

        logits = self.classifier(self.final_norm(x))
        return logits, lengths
