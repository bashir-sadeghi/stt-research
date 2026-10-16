# Transformer–CTC speech recognition

Speech-to-text adaptation of [Bashir Sadeghi's SLM framework](https://github.com/bashir-sadeghi/SLM), using its configuration, DataModule, PyTorch model, Lightning controller, logging, and checkpoint organization.

**Status: development in progress.** The two-layer LibriSpeech baseline has 27 completed Colab epochs in the user-run logs. The approved four-layer capacity experiment is configured but has not run. The AI has reviewed source and logs without launching training or tests. Checkpoint reproduction and final test-set evaluation remain unverified. Dependency pins are not a validated environment lock.

The selected architecture is trained from random initialization:

```text
Audio → log-Mel features → CNN downsampling → Transformer encoder
      → character logits → greedy CTC decoding → text
```

CTC loss trains the model without frame-level transcript alignments. Greedy decoding is separate from the loss. The offline encoder uses future context. [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

**Current experiment:** does increasing Transformer encoder depth from two to four layers improve transcription accuracy at matched training-data exposure? Width remains 128. The previously written masking comparison was an AI proposal incorrectly presented as agreed; masking is not part of this experiment.

## Review these files

| File | Responsibility |
| --- | --- |
| [lab/models/net.py](lab/models/net.py) | CNN, positional encoding, Transformer blocks, character logits, output lengths |
| [lab/datasets/librispeech.py](lab/datasets/librispeech.py) | Audio/transcripts, log-Mel normalization, character vocabulary, padded batches, greedy CTC decoding |
| [lab/losses/ctc.py](lab/losses/ctc.py) | Differentiable CTC loss and detached epoch loss statistics |
| [control/stt.py](control/stt.py) | Lightning training/validation/test hooks, manual optimization, WER/CER, transcript logging |
| [exps/stt.ini](exps/stt.ini), [exps/stt_colab.ini](exps/stt_colab.ini) | Local and Colab experiment profiles |
| [main.py](main.py), [config.py](config.py) | Your framework's experiment entry point, INI/CLI parsing, Trainer, and checkpoints |

Generic framework helpers remain under their original `lab/` locations. There is no parallel package, custom training engine, separate CLI, test suite, debugging configuration, notebook, or bundle tool.

## Batch and model contracts

The DataModule returns a dictionary with `features` `[B,T,80]`, `feature_lengths` `[B]`, concatenated character `targets`, `target_lengths` `[B]`, reference `texts`, utterance `ids`, `speakers`, and `durations`.

The model returns `logits` `[B,T_out,29]` and valid `output_lengths` `[B]`. Two stride-2 CNN layers reduce time by approximately four; exact lengths use ceiling division. Padding is masked between convolutions and in attention.

The vocabulary is CTC blank (ID 0), space, apostrophe, and uppercase A–Z. Transcript normalization is explicit in `CharacterTokenizer`. Greedy decoding merges consecutive repeated labels before removing blanks; decoding uses only valid output frames. Audio and transcript lengths are different and both are passed to CTC.

Features are prepared on CPU before batching. The controller moves acoustic features to the model device. The loss wrapper explicitly uses CPU CTC for MPS, preserving autograd through the transfer. User-run two-layer MPS and CUDA training have exercised these paths; the four-layer profile remains unexecuted. No global fallback is configured.

WER/CER use accumulated corpus edit counts through TorchMetrics. The returned training loss stays connected for `manual_backward`; only logging statistics are detached. TensorBoard text examples contain reference and predicted transcripts.

## Setup and execution

Run commands from the repository root in an environment with compatible dependencies. Existing local training uses Python 3.12. Version pins in [requirements.txt](requirements.txt) are not a complete environment lock.

```bash
python -m pip install -r requirements.txt
```

The local profile retains two encoder layers; the Colab profile now configures the four-layer experiment. Both use width 128 and the same dataset splits, optimizer, scheduler, seed, precision, and configured epoch count. Their settings are:

| Setting | Local: `exps/stt.ini` | Colab: `exps/stt_colab.ini` |
| --- | --- | --- |
| Encoder layers | 2 | 4 |
| Training batch | 2 | 4 |
| Gradient accumulation | 2 | 1 |
| Effective training batch (full groups) | 4 | 4 |
| Validation batch | 2 | 4 |
| Data root | `./data` | `/content/data` |
| Results root | `./results` | `/content/drive/MyDrive/stt-research/results` |

The comparison baseline is the earlier **two-layer Colab run**, which has 27 logged epochs. Compare the common first 27 epochs with the four-layer run; the configured maximum remains 30. Equal epochs control data exposure, while the deeper model uses more computation.

For local training:

```bash
python main.py --config exps/stt.ini
```

For Colab, select a GPU runtime, place the repository in the runtime, install compatible dependencies, and mount Google Drive before launching. In a notebook cell:

```python
import torch
from google.colab import drive

assert torch.cuda.is_available(), "Select a Colab GPU runtime first."
drive.mount("/content/drive")
```

Then run this command from the repository root in Colab (prefix with `!` in a notebook cell):

```bash
python main.py --config exps/stt_colab.ini
```

The GPU check and Drive mount are prerequisites: the current entry point otherwise falls back to CPU when no GPU is available, and an unmounted Drive path can be created as an ordinary temporary directory. The four-layer Colab profile is unexecuted; its memory fit and performance are unknown. Colab runtimes are temporary and hardware availability varies. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Both profiles currently enable `dataset_options["download"]`. Extracted audio lives at `<features_path>/LibriSpeech/<split>/`; the Colab data root is on the temporary runtime disk, while checkpoints, TensorBoard events, and saved hyperparameters go under the configured results root on mounted Drive. Set `download` to `false` when providing extracted data. The official TorchAudio downloader handles the requested full LibriSpeech splits; no pretrained model is used. [Dataset and license](https://www.openslr.org/12), [official dataset loader](https://docs.pytorch.org/audio/2.8/generated/torchaudio.datasets.LIBRISPEECH.html).

Ordinary training uses only the training/development splits and selects the lowest validation loss, following the framework's checkpoint convention. Test evaluation is a separate explicit command after development choices are frozen:

```bash
python main.py --config exps/stt.ini --test-only True --resume PATH_TO_CHECKPOINT
```

Use the same model dimensions/profile as the checkpoint when evaluating. Start the new four-layer experiment from random initialization, without `--resume`.

`--resume` during training retains the framework's existing weights-only behavior; full optimizer/scheduler/epoch restoration remains deferred. This matters for Colab session interruptions: the current option is not a complete training-state resume. MPS/CUDA/CPU selection follows the framework (`ngpu=0` selects CPU). Four-layer training, multi-device execution, and final convergence remain unverified.

## Research record

- [Research notes](docs/research-notes.md)
- [Decision log](docs/decision-log.md)

The local research record separates historical observations from the removed implementation and actual user-run outcomes from this Lightning adaptation. No completed 30-epoch run or final test-set result is available. The user handles Git commits and GitHub publication.
