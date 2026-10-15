"""LibriSpeech utterances, CPU log-Mel features, and character CTC decoding."""

from pathlib import Path
import random
import re
import unicodedata

import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
import torchaudio

__all__ = ["LibriSpeech"]


class CharacterTokenizer:
    """A fixed English character vocabulary; blank is not a transcript token."""

    tokens = ("<blank>", " ", "'", *tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    blank_id = 0
    vocab_size = len(tokens)

    def __init__(self):
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = dict(enumerate(self.tokens))

    @staticmethod
    def normalize(text):
        text = unicodedata.normalize("NFKC", text).upper().replace("’", "'")
        text = re.sub(r"[^A-Z' ]", " ", text)
        return " ".join(text.split())

    def encode(self, text):
        text = self.normalize(text)
        if not text:
            raise ValueError("A CTC transcript must contain at least one character")
        return torch.tensor([self.stoi[char] for char in text], dtype=torch.long)

    def greedy_decode(self, logits, lengths):
        """Select framewise argmax, merge consecutive repeats, then remove blank."""
        if logits.ndim != 3 or logits.shape[-1] != self.vocab_size:
            raise ValueError("Expected character logits with shape [batch, frames, 29]")
        lengths = torch.as_tensor(lengths).detach().cpu().tolist()
        if len(lengths) != logits.shape[0]:
            raise ValueError("Expected one valid output length per utterance")
        paths = logits.detach().argmax(dim=-1).cpu().tolist()
        hypotheses = []
        for path, length in zip(paths, lengths):
            if length < 1 or length > len(path):
                raise ValueError("Invalid CTC output length")
            previous = None
            characters = []
            for token_id in path[:length]:
                if token_id != previous and token_id != self.blank_id:
                    characters.append(self.itos[token_id])
                # A blank separates repeated letters: L, blank, L decodes to LL.
                previous = token_id
            hypotheses.append("".join(characters).strip())
        return hypotheses


class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("This LibriSpeech pipeline expects sample_rate=16000")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0,
            mel_scale="htk",
            norm=None,
        )

    def forward(self, waveform):
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("Expected a nonempty mono waveform")
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains nonfinite samples")
        if waveform.numel() < self.n_fft:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_fft - waveform.numel()))
        features = self.mel(waveform).clamp_min(1e-10).log().transpose(0, 1)
        # Normalize each utterance before padding, so padding does not affect statistics.
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, correction=0, keepdim=True).clamp_min(1e-5)
        return ((features - mean) / std).contiguous()


class SpeechDataset(Dataset):
    """Read transcript metadata eagerly and decode each audio file on demand."""

    def __init__(self, root, split, tokenizer, feature_options):
        self.split = split
        self.tokenizer = tokenizer
        self.features = LogMel(**feature_options)
        split_root = Path(root) / "LibriSpeech" / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Missing LibriSpeech split: {split_root}. "
                "Provide the extracted dataset or enable dataset_options.download."
            )
        self.records = []
        self.utterance_ids = set()
        self.speakers = set()
        self._audio_lengths = None
        for transcript_path in sorted(split_root.rglob("*.trans.txt")):
            for line_number, line in enumerate(transcript_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    raise ValueError(f"Missing transcript at {transcript_path}:{line_number}")
                utterance_id, text = fields
                text = tokenizer.normalize(text)
                if not text:
                    raise ValueError(f"Empty normalized transcript for {utterance_id}")
                if utterance_id in self.utterance_ids:
                    raise ValueError(f"Duplicate utterance ID in {split}: {utterance_id}")
                path = transcript_path.parent / f"{utterance_id}.flac"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing audio file: {path}")
                speaker = utterance_id.split("-", 1)[0]
                self.records.append({"path": path, "text": text, "id": utterance_id, "speaker": speaker})
                self.utterance_ids.add(utterance_id)
                self.speakers.add(speaker)
        if not self.records:
            raise ValueError(f"No LibriSpeech utterances found in {split_root}")

    def __len__(self):
        return len(self.records)

    def audio_lengths(self):
        """Cache sample counts from audio headers without decoding waveforms."""
        if self._audio_lengths is None:
            lengths = []
            for record in self.records:
                info = sf.info(str(record["path"]))
                if (info.samplerate != self.features.sample_rate
                        or info.channels != 1 or info.frames <= 0):
                    raise ValueError(f"Expected nonempty mono 16 kHz audio: {record['path']}")
                lengths.append(info.frames)
            self._audio_lengths = lengths
        return self._audio_lengths

    def __getitem__(self, index):
        record = self.records[index]
        waveform, sample_rate = sf.read(record["path"], dtype="float32")
        if sample_rate != self.features.sample_rate or waveform.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz audio: {record['path']}")
        waveform = torch.from_numpy(waveform)
        return {
            "features": self.features(waveform),
            "targets": self.tokenizer.encode(record["text"]),
            "text": record["text"],
            "id": record["id"],
            "speaker": record["speaker"],
            "duration": waveform.numel() / sample_rate,
        }


def collate_speech(items):
    features = [item["features"] for item in items]
    targets = [item["targets"] for item in items]
    return {
        "features": pad_sequence(features, batch_first=True, padding_value=0.0),
        "feature_lengths": torch.tensor([value.shape[0] for value in features], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor([value.numel() for value in targets], dtype=torch.long),
        "texts": [item["text"] for item in items],
        "ids": [item["id"] for item in items],
        "speakers": [item["speaker"] for item in items],
        "durations": torch.tensor([item["duration"] for item in items], dtype=torch.float32),
    }


class DurationGroupedSampler(Sampler):
    """Shuffle batches of similar-length utterances without dropping examples."""

    def __init__(self, lengths, batch_size, seed, pool_batches=100):
        if batch_size < 1 or pool_batches < 1:
            raise ValueError("Batch size and pool_batches must be positive")
        self.lengths = tuple(lengths)
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_batches
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch):
        # Lightning calls this on the DataLoader's sampler each epoch.
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1  # Also reshuffle when iterating outside Lightning.
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        remainder = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start:start + self.pool_size]
            pool.sort(key=self.lengths.__getitem__)
            for offset in range(0, len(pool), self.batch_size):
                batch = pool[offset:offset + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
                else:
                    remainder = batch
        rng.shuffle(batches)
        for batch in batches:
            yield from batch
        # Keep the partial batch last so DataLoader preserves batch boundaries.
        yield from remainder


class LibriSpeech(pl.LightningDataModule):
    official_splits = {
        "train-clean-100", "train-clean-360", "train-other-500",
        "dev-clean", "dev-other", "test-clean", "test-other",
    }

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.root = Path(opts.features_path).expanduser()
        self.tokenizer = CharacterTokenizer()
        options = opts.dataset_options
        self.feature_options = dict(options["feature_options"])
        if opts.model_options["vocab_size"] != self.tokenizer.vocab_size:
            raise ValueError("model_options.vocab_size must equal the 29-character CTC vocabulary")
        if opts.model_options["n_mels"] != self.feature_options["n_mels"]:
            raise ValueError("Model n_mels must match dataset feature_options.n_mels")
        self.train_split = options.get("train_split", "train-clean-100")
        self.dev_split = options.get("dev_split", "dev-clean")
        self.test_split = options.get("test_split", "test-clean")
        self.download = options.get("download", False)
        self.duration_batching = options.get("duration_batching", False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not self.download:
            return
        splits = [self.test_split] if self.opts.test_only else [self.train_split, self.dev_split]
        for split in dict.fromkeys(splits):
            if split not in self.official_splits:
                raise ValueError(
                    f"Official torchaudio LibriSpeech download does not support split {split!r}. "
                    "Existing extracted splits may be read with download=false."
                )
        self.root.mkdir(parents=True, exist_ok=True)
        for split in dict.fromkeys(splits):
            torchaudio.datasets.LIBRISPEECH(root=str(self.root), url=split, download=True)

    def _dataset(self, split):
        return SpeechDataset(self.root, split, self.tokenizer, self.feature_options)

    def _check_train_dev(self):
        if self.train_dataset is None or self.val_dataset is None:
            return
        if self.train_dataset.utterance_ids & self.val_dataset.utterance_ids:
            raise ValueError("Training and validation splits contain overlapping utterance IDs")
        if self.train_dataset.speakers & self.val_dataset.speakers:
            raise ValueError("Training and validation splits contain overlapping speakers")

    def setup(self, stage=None):
        if stage is None:
            stage = "test" if self.opts.test_only else "fit"
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self._dataset(self.train_split)
            if self.val_dataset is None:
                self.val_dataset = self._dataset(self.dev_split)
            self._check_train_dev()
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._dataset(self.dev_split)
            self._check_train_dev()
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._dataset(self.test_split)
        else:
            raise ValueError(f"Unsupported LibriSpeech setup stage: {stage!r}")

    def _loader(self, dataset, batch_size, shuffle=False):
        if dataset is None:
            raise RuntimeError("Call setup() for the requested stage before creating its loader")
        sampler = None
        if shuffle and self.duration_batching:
            sampler = DurationGroupedSampler(
                dataset.audio_lengths(), batch_size, self.opts.manual_seed,
            )
            shuffle = False
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.opts.nthreads,
            pin_memory=torch.cuda.is_available() and self.opts.ngpu != 0,
            collate_fn=collate_speech,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.opts.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.opts.batch_size_val)

    def test_dataloader(self):
        return self._loader(self.test_dataset, self.opts.batch_size_test)

    def test_train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = self._dataset(self.train_split)
        return self._loader(self.train_dataset, self.opts.batch_size_test)
