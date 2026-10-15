# Transformer–CTC speech recognition

Speech-to-text adaptation of [Bashir Sadeghi's SLM framework](https://github.com/bashir-sadeghi/SLM), using its configuration, DataModule, PyTorch model, Lightning controller, logging, and checkpoint organization.

**Status: implementation draft awaiting user feedback. No tests, imports, training, or evaluation have been run on this replacement.** Historical checks of the previous implementation do not validate the adapted Lightning pipeline. Dependencies are specified but have not been installed or compatibility-tested for this replacement.

The selected architecture is trained from random initialization:

```text
Audio → log-Mel features → CNN downsampling → Transformer encoder
      → character logits → greedy CTC decoding → text
```

CTC loss trains the model without frame-level transcript alignments. Greedy decoding is separate from the loss. The offline encoder uses future context. [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

The specific research question and second experimental approach are **unconfirmed**. The previously written masking comparison was an AI proposal incorrectly presented as agreed; it is not part of this draft.

## Review these files

| File | Responsibility |
| --- | --- |
| [lab/models/net.py](lab/models/net.py) | CNN, positional encoding, Transformer blocks, character logits, output lengths |
| [lab/datasets/librispeech.py](lab/datasets/librispeech.py) | Audio/transcripts, log-Mel normalization, character vocabulary, padded batches, greedy CTC decoding |
| [lab/losses/ctc.py](lab/losses/ctc.py) | Differentiable CTC loss and detached epoch loss statistics |
| [control/stt.py](control/stt.py) | Lightning training/validation/test hooks, manual optimization, WER/CER, transcript logging |
| [exps/stt.ini](exps/stt.ini) | Provisional model, dataset, batch, and optimizer settings |
| [main.py](main.py), [config.py](config.py) | Your framework's experiment entry point, INI/CLI parsing, Trainer, and checkpoints |

Generic framework helpers remain under their original `lab/` locations. There is no parallel package, custom training engine, separate CLI, test suite, debugging configuration, notebook, or bundle tool.

## Batch and model contracts

The DataModule returns a dictionary with `features` `[B,T,80]`, `feature_lengths` `[B]`, concatenated character `targets`, `target_lengths` `[B]`, reference `texts`, utterance `ids`, `speakers`, and `durations`.

The model returns `logits` `[B,T_out,29]` and valid `output_lengths` `[B]`. Two stride-2 CNN layers reduce time by approximately four; exact lengths use ceiling division. Padding is masked between convolutions and in attention.

The vocabulary is CTC blank (ID 0), space, apostrophe, and uppercase A–Z. Transcript normalization is explicit in `CharacterTokenizer`. Greedy decoding merges consecutive repeated labels before removing blanks; decoding uses only valid output frames. Audio and transcript lengths are different and both are passed to CTC.

Features are prepared on CPU before batching. The controller moves acoustic features to the model device. The loss wrapper explicitly uses CPU CTC for MPS, preserving autograd through the transfer. This behavior is ported from the previous implementation; its Lightning integration has not been verified. No global fallback is configured.

WER/CER use accumulated corpus edit counts through TorchMetrics. The returned training loss stays connected for `manual_backward`; only logging statistics are detached. TensorBoard text examples contain reference and predicted transcripts.

## Setup and execution, after feedback

Run these commands from the repository root only after feedback. They are documentation only and have not been executed for this replacement. Python 3.12 is the intended interpreter. Version pins in [requirements.txt](requirements.txt) are draft dependencies, not a tested environment lock.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --config exps/stt.ini
```

The config defaults to `download: false`. Provide extracted LibriSpeech under `data/LibriSpeech/<split>/`, or explicitly set `dataset_options["download"]` to `true` when ready. The official TorchAudio downloader handles the requested full LibriSpeech splits; no pretrained model is used. [Dataset and license](https://www.openslr.org/12), [official dataset loader](https://docs.pytorch.org/audio/2.8/generated/torchaudio.datasets.LIBRISPEECH.html).

Ordinary training uses only the training/development splits and selects the lowest validation loss, following the framework's checkpoint convention. Test evaluation is a separate explicit command after development choices are frozen:

```bash
python main.py --config exps/stt.ini --test-only True --resume PATH_TO_CHECKPOINT
```

`--resume` during training retains the framework's existing weights-only behavior; full optimizer/scheduler resume is deferred to the user. MPS/CUDA/CPU selection follows the framework (`ngpu=0` selects CPU). Multi-device execution, backend compatibility, convergence, and runtime are unverified for this draft.

## Research record

- [Research notes](docs/research-notes.md)
- [Decision log](docs/decision-log.md)

Prior measurements are historical observations from the removed implementation, not outcomes of this replacement. No full training or final test-set result is available. The user handles Git commits and GitHub publication.
