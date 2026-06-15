# Quantum Seuss

Quantum Seuss is a tiny local hybrid quantum-classical language model.

It trains a miniature word-level Transformer on authorized local corpus text,
optionally routes hidden states through a simulated PennyLane quantum circuit,
and samples generated tokens with quantum random numbers.

This is a playful research toy, not a general-purpose chatbot.

## What Is Quantum?

The model is hybrid quantum-classical:

- **Classical backbone:** PyTorch token embeddings, Transformer blocks, optimizer, and logits.
- **Quantum layer:** optional trainable PennyLane variational quantum circuit inside the network.
- **Quantum initialization:** optional ANU QRNG seed for model initialization.
- **Quantum mixer:** optional QRNG-seeded orthogonal feature mixer.
- **Quantum sampling:** default ANU QRNG-backed token sampling during generation.

The quantum circuit uses PennyLane's local `default.qubit` simulator unless you
replace it with another backend. So the model is quantum-structured and
quantum-randomized, but it is not running on quantum hardware by default.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies include PyTorch for the Transformer and PennyLane for the optional
trainable quantum circuit layer.

For OCR from FlipHTML5 page images, install Tesseract:

```bash
brew install tesseract
```

Generate corpus text from the example authorized FlipHTML5 URL manifest:

```bash
cp examples/fliphtml5_urls.example.txt fliphtml5_urls.txt
python scripts/ingest_fliphtml5.py --manifest fliphtml5_urls.txt --out corpus --clean
```

Train a hybrid quantum-classical model:

```bash
python llm.py train \
  --corpus corpus \
  --steps 600 \
  --log-every 100 \
  --batch-size 8 \
  --block-size 24 \
  --embedding-size 64 \
  --heads 4 \
  --layers 2 \
  --quantum-init \
  --quantum-layer \
  --quantum-qubits 4 \
  --quantum-circuit-layers 1 \
  --out quantum_seuss_llm.pt
```

Generate with QRNG-backed sampling:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "a sleepy bug" \
  --tokens 120 \
  --temperature 0.8 \
  --top-k 12
```

## Architecture

```text
tokens
→ token + position embeddings
→ causal Transformer encoder blocks
→ optional PennyLane quantum residual layer
→ optional QRNG-seeded orthogonal mixer
→ layer norm + logits
→ QRNG/token sampler
```

The trainable quantum residual layer works like this:

```text
hidden state
→ linear projection to n qubits
→ AngleEmbedding
→ StronglyEntanglingLayers
→ PauliZ expectation values
→ linear projection back to hidden size
→ residual add
```

Use small settings first. Quantum simulation is much slower than plain PyTorch,
especially because it runs over token positions.

## Corpus Workflow

This repo does **not** package Dr. Seuss books, OCR output, raw source files, or
trained checkpoints. Those are generated locally and ignored by `.gitignore`.

### Local Files

Put licensed or otherwise authorized files in `corpus_raw/`.

Supported formats:

- `.txt`
- `.md`
- `.html` / `.htm`
- `.pdf`
- `.epub`

Extract them into `corpus/`:

```bash
python scripts/prepare_corpus.py corpus_raw --clean
```

### Authorized File URLs

Create `licensed_urls.txt` with one authorized download URL per line:

```bash
python scripts/prepare_corpus.py --manifest licensed_urls.txt --clean
```

### Authorized FlipHTML5 Mirrors

Copy the example manifest:

```bash
cp examples/fliphtml5_urls.example.txt fliphtml5_urls.txt
```

OCR every URL in the manifest:

```bash
python scripts/ingest_fliphtml5.py --manifest fliphtml5_urls.txt --out corpus --clean
```

OCR a single mirror:

```bash
python scripts/ingest_fliphtml5.py "https://online.fliphtml5.com/piaso/fncp/#p=8" \
  --name dr-seuss-book-of-bedtime-stories \
  --out corpus \
  --keep-images \
  --clean
```

## Training Examples

Plain tiny Transformer:

```bash
python llm.py train \
  --corpus corpus \
  --steps 1400 \
  --log-every 200 \
  --batch-size 48 \
  --block-size 48 \
  --embedding-size 128 \
  --heads 4 \
  --layers 3 \
  --out seuss_llm.pt
```

QRNG-seeded classical model:

```bash
python llm.py train \
  --corpus corpus \
  --steps 1400 \
  --log-every 200 \
  --batch-size 48 \
  --block-size 48 \
  --embedding-size 128 \
  --heads 4 \
  --layers 3 \
  --quantum-init \
  --quantum-mixer \
  --out seuss_llm.pt
```

Hybrid model with trainable quantum circuit:

```bash
python llm.py train \
  --corpus corpus \
  --steps 600 \
  --log-every 100 \
  --batch-size 8 \
  --block-size 24 \
  --embedding-size 64 \
  --heads 4 \
  --layers 2 \
  --quantum-init \
  --quantum-layer \
  --quantum-qubits 4 \
  --quantum-circuit-layers 1 \
  --out quantum_seuss_llm.pt
```

Useful training flags:

- `--quantum-init`: seed model initialization from ANU QRNG.
- `--quantum-mixer`: add a fixed QRNG-seeded orthogonal hidden-state mixer.
- `--quantum-layer`: add a trainable PennyLane quantum residual layer.
- `--quantum-qubits`: number of simulated qubits for the quantum layer.
- `--quantum-circuit-layers`: number of variational circuit blocks.
- `--strict-quantum`: fail instead of falling back if QRNG is unavailable.

## Generation Examples

Generation prepends this default startup instruction:

```text
you are a dr seuss book. write playful rhyming children story prose with whimsical creatures, simple words, repetition, and a bouncy rhythm.
```

The instruction conditions generation but is hidden from printed output.

Balanced:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "a sleepy bug" \
  --tokens 120 \
  --temperature 0.8 \
  --top-k 12
```

More coherent/repetitive:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "i do not" \
  --tokens 120 \
  --temperature 0.55 \
  --top-k 6
```

Tiny model, big question:

```bash
python llm.py generate \
  --model seuss_llm.pt \
  --prompt "What is the meaning of life" \
  --tokens 120 \
  --temperature 0.55 \
  --top-k 6
```

More chaotic/creative:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "once upon" \
  --tokens 120 \
  --temperature 0.95 \
  --top-k 18
```

Require QRNG sampling:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "a sleepy bug" \
  --sampler anu-qrng \
  --strict-quantum
```

Use deterministic PyTorch sampling for debugging:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "a sleepy bug" \
  --sampler torch \
  --seed 7
```

Override the startup instruction:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --instruction "you are a bouncy rhyming bedtime book" \
  --prompt "the moon had shoes"
```

Disable the startup instruction:

```bash
python llm.py generate \
  --model quantum_seuss_llm.pt \
  --prompt "horton" \
  --no-instruction
```

## Source Tracking

Build a source/acquisition checklist:

```bash
python scripts/build_source_manifest.py
```

This writes `sources.json` with official catalog/source-page records. Use it as
provenance tracking for licensed or authorized text exports.

## Files

- `llm.py`: tiny Transformer, PennyLane quantum layer, training, and generation.
- `corpus/`: generated cleaned training text; only `README.md` is tracked.
- `corpus_raw/`: generated/raw source files; only `README.md` is tracked.
- `examples/fliphtml5_urls.example.txt`: example authorized FlipHTML5 URL manifest.
- `scripts/prepare_corpus.py`: local file and URL-manifest ingestion.
- `scripts/ingest_fliphtml5.py`: OCR ingestion for authorized FlipHTML5 mirrors.
- `scripts/build_source_manifest.py`: official source checklist builder.
- `requirements.txt`: Python dependencies.

## Notes

- Generated text quality depends heavily on corpus size and OCR quality.
- Corpus text and model checkpoints are intentionally not packaged.
- The LLM is trained from scratch; it does not understand instructions like a chat model.
- PennyLane simulates the quantum circuit unless you configure another backend.
- QRNG entropy can affect initialization, hidden-state mixing, and token sampling.
- `temperature` controls randomness; lower is safer, higher is weirder.
- `top-k` limits sampling to the most likely next tokens.
