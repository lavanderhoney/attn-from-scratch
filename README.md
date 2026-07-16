# attn-from-scratch

Transformer-based English-to-French translation model implemented from first principles in PyTorch. The project includes a custom training pipeline, BPE tokenization, greedy and beam-search decoding, TensorBoard logging, checkpoint resume support, and a lightweight FastAPI demo wrapper for generation.

The goal of the repository is to show a full machine learning engineering workflow rather than just a model definition: data preparation, tokenization, training, inference, experiment tracking, and artifact management are all represented in the codebase.

## What This Project Does

- Trains a sequence-to-sequence Transformer on the `Helsinki-NLP/opus_books` English-French dataset.
- Builds separate BPE tokenizers for source and target languages.
- Uses a standard encoder-decoder Transformer stack with multi-head attention, feed-forward blocks, residual connections, and sinusoidal positional encoding.
- Supports teacher-forced training, greedy decoding, and beam-search decoding.
- Saves checkpoints with model, optimizer, scheduler, and RNG state so training can resume cleanly.
- Logs losses, learning rate, gradients, and parameter histograms to TensorBoard.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [src/attention.py](src/attention.py) | Scaled dot-product multi-head attention implementation. |
| [src/layers.py](src/layers.py) | Embeddings, positional encoding, feed-forward block, and residual wrapper. |
| [src/encoder.py](src/encoder.py) | Transformer encoder stack. |
| [src/decoder.py](src/decoder.py) | Transformer decoder stack. |
| [src/model.py](src/model.py) | Full encoder-decoder Transformer, masking, greedy decoding, and beam search. |
| [training/dataset.py](training/dataset.py) | Hugging Face dataset loading, BPE tokenizer handling, tokenization, and dataloaders. |
| [training/train.py](training/train.py) | Training loop, checkpoint saving, resume support, and TensorBoard logging. |
| [training/inference.py](training/inference.py) | Checkpoint loading and translation generation. |
| [backend/main.py](backend/main.py) | Minimal FastAPI demo for generation requests. |
| [training/config.yml](training/config.yml) | Base training and model configuration. |
| [training/config2.yml](training/config2.yml) | Continuation config for resuming training from a saved checkpoint. |
| [tokenizers/](tokenizers/) | Saved BPE tokenizers for English and French. |
| [checkpoints/](checkpoints/) | Saved model checkpoints from prior runs. |
| [runs/](runs/) | TensorBoard experiment logs. |

## Architecture Overview

The model follows the classic Transformer encoder-decoder design:

- Token embeddings are scaled by `sqrt(d_model)`.
- Sinusoidal positional encodings are added to the embeddings.
- The encoder and decoder each stack `N` blocks.
- Each encoder block contains self-attention and a feed-forward sublayer.
- Each decoder block contains masked self-attention, cross-attention, and a feed-forward sublayer.
- Residual connections use pre-layer normalization.
- The final decoder output is projected to vocabulary logits.

The implementation also includes:

- source padding masks for encoder and cross-attention,
- target padding masks plus causal masking for autoregressive decoding,
- Xavier initialization for parameters with more than one dimension,
- greedy decoding for fast generation,
- beam search for higher-quality generation.

## Data And Tokenization

The dataset is loaded from Hugging Face Datasets:

```text
Helsinki-NLP/opus_books, en-fr split
```

Tokenization is handled with separate Byte Pair Encoding tokenizers for English and French. The tokenizers use the following special tokens:

- `[UNK]`
- `[PAD]`
- `[SOS]`
- `[EOS]`

The dataset pipeline:

- loads the OPUS Books parallel corpus,
- tokenizes source and target text separately,
- truncates each side to `seq_len - 2`,
- wraps each sequence with start and end tokens,
- dynamically pads batches to the longest sequence in the batch.

## Training Details

The base configuration in [training/config.yml](training/config.yml) uses:

- `seq_len: 128`
- `d_model: 512`
- `d_ff: 2048`
- `num_heads: 8`
- `num_layers: 6`
- `vocab_size: 32000`
- `batch_size: 16`
- `n_epochs: 20`
- `lr: 1`
- `warmup: 4000`
- `test_size: 0.1`

Training uses:

- teacher forcing with right-shifted decoder inputs,
- cross-entropy loss with label smoothing,
- gradient clipping at `max_norm=1.0`,
- an Adam optimizer,
- a Noam-style learning-rate schedule,
- checkpoint saving every 2 epochs and at the final epoch.

Checkpoints include:

- model weights,
- optimizer state,
- scheduler state,
- epoch and global step counters,
- CPU and GPU RNG state,
- the configuration used for the run.

## Inference And Decoding

The model supports two decoding strategies:

- Greedy decoding, which is faster and simple to inspect.
- Beam-search decoding, which keeps multiple hypotheses alive during generation.

The inference script loads a checkpoint, reconstructs the model from the saved config, and then generates translations from either a user-provided sentence or a sample batch from the validation split.

## API Demo

The repository also contains a small FastAPI app in [backend/main.py](backend/main.py) intended as a minimal generation demo. It is useful as a starting point for turning the model into a service, but it is not production-hardened.

To load the checkpoint from Hugging Face instead of a local file, set these environment variables before starting the backend:

```bash
export CHECKPOINT_REPO_ID="your-username/your-model-repo"
export CHECKPOINT_FILENAME="transformer_noam_v2_epoch_30.pt"
# optional
export CHECKPOINT_REVISION="main"
```

If you still want to use a local checkpoint, set `CHECKPOINT_PATH` instead. The backend will prefer the local path when it exists.

For local development, you can put those variables in a repo-root `.env` file and start the backend normally. The app loads that file on startup.

## Requirements

The codebase targets Python 3.10+ and uses the following core libraries:

- PyTorch
- Hugging Face `datasets`
- `tokenizers`
- `PyYAML`
- `tqdm`
- `tensorboard`
- `fastapi`
- `uvicorn`

Install the dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install torch datasets tokenizers pyyaml tqdm tensorboard fastapi uvicorn
```

If you want GPU support, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Quick Start

### 1. Clone And Set Up The Environment

```bash
git clone <repo-url>
cd attn-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install torch datasets tokenizers pyyaml tqdm tensorboard fastapi uvicorn
```

### 2. Train The Model

```bash
python training/train.py --config training/config.yml --model-artifact-name transformer_noam_v2
```

This will:

- download the OPUS Books dataset on first run,
- load or build the English and French tokenizers,
- train the Transformer,
- write checkpoints to `checkpoints/`,
- log TensorBoard metrics under `runs/`.

### 3. Resume Training

Use [training/config2.yml](training/config2.yml) as an example of a continuation config:

```bash
python training/train.py --config training/config2.yml --model-artifact-name transformer_noam_v2
```

The resume path inside that config points to the saved epoch 20 checkpoint.

### 4. Run Inference

The inference module loads a checkpoint and generates translations:

```bash
python training/inference.py
```

The current script version hardcodes the checkpoint path in its `__main__` block. If you want to use a different checkpoint, update that path or import `load_model` and `generate_target` from another script.

### 5. Start TensorBoard

```bash
tensorboard --logdir runs
```

## Checkpoints And Artifacts

The repository already contains several trained checkpoints under [checkpoints/](checkpoints/), including epochs 18, 19, 20, 28, and 30. TensorBoard run folders are stored under [runs/](runs/).

The tokenizer JSON files are stored in [tokenizers/](tokenizers/). The training code expects those files to exist at the paths defined in the config.

## Current Caveats

The project is functional as a research and portfolio codebase, but a few parts are still best treated as demo-quality:

- The FastAPI route in [backend/main.py](backend/main.py) is a thin wrapper and should be tightened before production use.
- The inference script currently hardcodes a checkpoint path in `__main__` instead of exposing a clean CLI.
- Some debug prints remain in the decoding code, so generation output is verbose by default.
- Beam search does not currently apply length normalization, so shorter sequences can be favored.

## Suggested Next Improvements

If you want to take this further for a stronger portfolio presentation, the most valuable additions would be:

- a proper command-line interface for inference and serving,
- automated tests for masking, tokenization, and decoding,
- evaluation metrics such as BLEU on a held-out test set,
- a cleaned-up API layer with request validation and async handling,
- a small demo UI or notebook showing side-by-side translations.

## License

No license file is currently included. Add one before publishing or open-sourcing the project.