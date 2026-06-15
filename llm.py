import argparse
import json
import math
import re
import secrets
import sys
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_CORPUS = Path("corpus")
DEFAULT_MODEL = Path("tiny_seuss_llm.pt")
DEFAULT_INSTRUCTION = (
    "you are a dr seuss book. write playful rhyming children story prose "
    "with whimsical creatures, simple words, repetition, and a bouncy rhythm."
)
ANU_QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php"


def read_corpus(path):
    if path.is_file():
        return path.read_text(encoding="utf-8")

    if path.is_dir():
        text_files = sorted(path.glob("*.txt"))
        if not text_files:
            raise SystemExit(f"error: no .txt files found in corpus directory {path}")

        return "\n\n".join(file.read_text(encoding="utf-8") for file in text_files)

    raise SystemExit(f"error: corpus path {path} does not exist")


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise SystemExit(
            "error: PyTorch is required for llm.py. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    return torch, nn, functional


def require_pennylane():
    try:
        import pennylane as qml
    except ImportError as error:
        raise SystemExit(
            "error: PennyLane is required for --quantum-layer. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    return qml


def tokenize(text):
    normalized = text.lower().replace("\u2018", "'").replace("\u2019", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*|[.!?]", normalized)


def detokenize(tokens):
    text = " ".join(token for token in tokens if token != "<unk>")
    text = (
        text.replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
    )
    return re.sub(r"\bi\b", "I", re.sub(r"\bi'", "I'", text))


def build_vocab(tokens):
    vocabulary = ["<unk>"] + sorted(set(tokens))
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    return vocabulary, token_to_index


def encode(tokens, token_to_index):
    unknown = token_to_index["<unk>"]
    return [token_to_index.get(token, unknown) for token in tokens]


def filter_known_tokens(tokens, token_to_index):
    known = []
    unknown = []
    for token in tokens:
        if token in token_to_index:
            known.append(token)
        else:
            unknown.append(token)
    return known, unknown


def format_unknown_token_warning(unknown_tokens):
    unique_tokens = list(dict.fromkeys(unknown_tokens))
    return "warning: ignoring unknown prompt tokens: " + ", ".join(unique_tokens)


class EntropySource:
    def __init__(self, backend, torch, strict_quantum=False):
        self.backend = backend
        self.torch = torch
        self.strict_quantum = strict_quantum
        self.cache = []
        self.warned = False

    def random_float(self):
        if self.backend == "torch":
            return float(self.torch.rand(()).item())
        if self.backend == "os":
            return secrets.randbits(64) / 2**64
        if self.backend == "anu-qrng":
            return self._anu_uint16() / 65536

        raise ValueError(f"unknown sampler backend: {self.backend}")

    def _anu_uint16(self):
        if not self.cache:
            self._refill_anu_cache()
        return self.cache.pop()

    def _refill_anu_cache(self):
        query = urllib.parse.urlencode({"length": 1024, "type": "uint16"})
        url = f"{ANU_QRNG_URL}?{query}"
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "quantum-seuss-llm/1.0"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            if self.strict_quantum:
                raise SystemExit(f"error: ANU QRNG request failed: {error}") from error
            if not self.warned:
                print(
                    f"warning: ANU QRNG unavailable; falling back to OS entropy ({error})",
                    file=sys.stderr,
                )
                self.warned = True
            self.cache = [secrets.randbits(16) for _ in range(1024)]
            return

        if not payload.get("success", True) or "data" not in payload:
            message = f"unexpected ANU QRNG response: {payload!r}"
            if self.strict_quantum:
                raise SystemExit(f"error: {message}")
            if not self.warned:
                print(f"warning: {message}; falling back to OS entropy", file=sys.stderr)
                self.warned = True
            self.cache = [secrets.randbits(16) for _ in range(1024)]
            return

        self.cache = [int(value) for value in payload["data"]]

    def choose_index(self, probabilities):
        threshold = self.random_float()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if threshold < cumulative:
                return index
        return len(probabilities) - 1


def quantum_seed(torch, strict_quantum=False):
    entropy = EntropySource("anu-qrng", torch, strict_quantum)
    seed = 0
    for _ in range(4):
        seed = (seed << 16) | entropy._anu_uint16()
    return seed % (2**63 - 1)


def build_quantum_mixer(embedding_size, torch, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    random_matrix = torch.randn(embedding_size, embedding_size, generator=generator)
    orthogonal_matrix, _ = torch.linalg.qr(random_matrix)
    return orthogonal_matrix


def make_quantum_residual_layer(embedding_size, qubits, circuit_layers, torch, nn):
    qml = require_pennylane()
    device = qml.device("default.qubit", wires=qubits)

    @qml.qnode(device, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(qubits))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]

    weight_shapes = {"weights": (circuit_layers, qubits, 3)}

    class QuantumResidualLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qubits = nn.Linear(embedding_size, qubits)
            self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes)
            self.from_qubits = nn.Linear(qubits, embedding_size)
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, hidden):
            batch_size, sequence_length, hidden_size = hidden.shape
            flattened = hidden.reshape(batch_size * sequence_length, hidden_size)
            quantum_inputs = torch.tanh(self.to_qubits(flattened)) * math.pi
            quantum_outputs = self.quantum(quantum_inputs)
            residual = self.from_qubits(quantum_outputs)
            residual = residual.reshape(batch_size, sequence_length, hidden_size)
            return hidden + self.scale * residual

    return QuantumResidualLayer()


class TinyTransformer:
    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_size,
        heads,
        layers,
        dropout,
        quantum_mixer=False,
        mixer_seed=0,
        quantum_layer=False,
        quantum_qubits=4,
        quantum_circuit_layers=1,
    ):
        torch, nn, functional = require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, embedding_size)
                self.position_embedding = nn.Embedding(block_size, embedding_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_size,
                    nhead=heads,
                    dim_feedforward=embedding_size * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.blocks = nn.TransformerEncoder(encoder_layer, layers)
                self.norm = nn.LayerNorm(embedding_size)
                self.output = nn.Linear(embedding_size, vocab_size)
                self.block_size = block_size
                if quantum_layer:
                    self.quantum_layer = make_quantum_residual_layer(
                        embedding_size,
                        quantum_qubits,
                        quantum_circuit_layers,
                        torch,
                        nn,
                    )
                else:
                    self.quantum_layer = None
                if quantum_mixer:
                    self.register_buffer(
                        "quantum_mixer",
                        build_quantum_mixer(embedding_size, torch, mixer_seed),
                    )
                else:
                    self.quantum_mixer = None

            def forward(self, input_ids, targets=None):
                batch_size, sequence_length = input_ids.shape
                positions = torch.arange(sequence_length, device=input_ids.device)
                hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
                mask = torch.triu(
                    torch.full(
                        (sequence_length, sequence_length),
                        float("-inf"),
                        device=input_ids.device,
                    ),
                    diagonal=1,
                )
                hidden = self.blocks(hidden, mask=mask)
                if self.quantum_layer is not None:
                    hidden = self.quantum_layer(hidden)
                if self.quantum_mixer is not None:
                    hidden = hidden @ self.quantum_mixer
                logits = self.output(self.norm(hidden))

                loss = None
                if targets is not None:
                    loss = functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )

                return logits, loss

        self.model = Model()


def make_batch(data, block_size, batch_size, torch):
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


def train(args):
    torch, nn, functional = require_torch()
    text = read_corpus(args.corpus)
    tokens = tokenize(text)
    if len(tokens) < args.block_size + 2:
        raise SystemExit("error: corpus is too small for the requested --block-size")

    vocabulary, token_to_index = build_vocab(tokens)
    data = torch.tensor(encode(tokens, token_to_index), dtype=torch.long)
    seed = args.seed
    mixer_seed = args.mixer_seed
    if args.quantum_init:
        seed = quantum_seed(torch, args.strict_quantum)
        print(f"quantum init seed {seed}", flush=True)
    if args.quantum_mixer and mixer_seed is None:
        mixer_seed = quantum_seed(torch, args.strict_quantum)
        print(f"quantum mixer seed {mixer_seed}", flush=True)

    torch.manual_seed(seed)

    wrapper = TinyTransformer(
        vocab_size=len(vocabulary),
        block_size=args.block_size,
        embedding_size=args.embedding_size,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        quantum_mixer=args.quantum_mixer,
        mixer_seed=mixer_seed or seed,
        quantum_layer=args.quantum_layer,
        quantum_qubits=args.quantum_qubits,
        quantum_circuit_layers=args.quantum_circuit_layers,
    )
    model = wrapper.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = make_batch(data, args.block_size, args.batch_size, torch)
        logits, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:04d} loss {loss.item():.4f}", flush=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "vocabulary": vocabulary,
        "config": {
            "block_size": args.block_size,
            "embedding_size": args.embedding_size,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "quantum_mixer": args.quantum_mixer,
            "mixer_seed": mixer_seed or seed,
            "quantum_layer": args.quantum_layer,
            "quantum_qubits": args.quantum_qubits,
            "quantum_circuit_layers": args.quantum_circuit_layers,
        },
    }
    torch.save(checkpoint, args.out)
    print(f"saved {args.out}", flush=True)


def sample_next_token(logits, temperature, top_k, torch, entropy, forbidden_ids=None):
    if forbidden_ids:
        logits = logits.clone()
        for token_id in forbidden_ids:
            logits[token_id] = float("-inf")

    logits = logits / max(temperature, 1e-6)
    if top_k:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = torch.where(logits < values[-1], torch.full_like(logits, float("-inf")), logits)

    probabilities = torch.softmax(logits, dim=-1)
    return entropy.choose_index(probabilities.tolist())


def generate(args):
    torch, nn, functional = require_torch()
    if not args.model.exists():
        raise SystemExit(
            f"error: model file {args.model} does not exist. Train one with "
            "`python llm.py train` first."
        )

    checkpoint = torch.load(args.model, map_location="cpu")
    vocabulary = checkpoint["vocabulary"]
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    forbidden_ids = [token_to_index["<unk>"]]
    config = checkpoint["config"]

    config.setdefault("quantum_mixer", False)
    config.setdefault("mixer_seed", 0)
    config.setdefault("quantum_layer", False)
    config.setdefault("quantum_qubits", 4)
    config.setdefault("quantum_circuit_layers", 1)
    wrapper = TinyTransformer(vocab_size=len(vocabulary), **config)
    model = wrapper.model
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    torch.manual_seed(args.seed)
    entropy = EntropySource(args.sampler, torch, args.strict_quantum)

    instruction_tokens = []
    prompt_tokens = tokenize(args.prompt)
    known_prompt_tokens, unknown_prompt_tokens = filter_known_tokens(prompt_tokens, token_to_index)
    if unknown_prompt_tokens:
        print(format_unknown_token_warning(unknown_prompt_tokens), file=sys.stderr)

    tokens = known_prompt_tokens
    if args.instruction and not args.no_instruction:
        instruction_tokens = [
            token for token in tokenize(args.instruction) if token in token_to_index
        ]
        tokens = instruction_tokens + known_prompt_tokens

    if not tokens:
        tokens = ["i"]
    generated = encode(tokens, token_to_index)

    with torch.no_grad():
        for _ in range(args.tokens):
            context = generated[-config["block_size"] :]
            input_ids = torch.tensor([context], dtype=torch.long)
            logits, _ = model(input_ids)
            next_token = sample_next_token(
                logits[0, -1],
                args.temperature,
                args.top_k,
                torch,
                entropy,
                forbidden_ids,
            )
            generated.append(next_token)

    visible_tokens = generated[len(instruction_tokens) :]
    print(detokenize([vocabulary[index] for index in visible_tokens]))


def parse_args():
    parser = argparse.ArgumentParser(description="Train or sample a tiny Transformer language model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--steps", type=int, default=800)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--block-size", type=int, default=32)
    train_parser.add_argument("--embedding-size", type=int, default=96)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--layers", type=int, default=3)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument(
        "--quantum-init",
        action="store_true",
        help="Use ANU QRNG entropy to seed model initialization.",
    )
    train_parser.add_argument(
        "--quantum-mixer",
        action="store_true",
        help="Insert a fixed QRNG-seeded orthogonal feature mixer after the Transformer blocks.",
    )
    train_parser.add_argument(
        "--quantum-layer",
        action="store_true",
        help="Insert a trainable PennyLane variational quantum circuit layer.",
    )
    train_parser.add_argument(
        "--quantum-qubits",
        type=int,
        default=4,
        help="Number of simulated qubits in --quantum-layer.",
    )
    train_parser.add_argument(
        "--quantum-circuit-layers",
        type=int,
        default=1,
        help="Number of StronglyEntanglingLayers blocks in --quantum-layer.",
    )
    train_parser.add_argument(
        "--mixer-seed",
        type=int,
        help="Optional explicit seed for the quantum mixer; otherwise QRNG/seed is used.",
    )
    train_parser.add_argument(
        "--strict-quantum",
        action="store_true",
        help="Fail instead of falling back if QRNG entropy is unavailable during quantum init/mixer setup.",
    )
    train_parser.set_defaults(handler=train)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    generate_parser.add_argument("--prompt", default="i do not")
    generate_parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    generate_parser.add_argument("--no-instruction", action="store_true")
    generate_parser.add_argument("--tokens", type=int, default=80)
    generate_parser.add_argument("--temperature", type=float, default=0.9)
    generate_parser.add_argument("--top-k", type=int, default=12)
    generate_parser.add_argument(
        "--sampler",
        choices=("anu-qrng", "torch", "os"),
        default="anu-qrng",
        help="Entropy source for token sampling. anu-qrng uses ANU quantum random numbers.",
    )
    generate_parser.add_argument(
        "--strict-quantum",
        action="store_true",
        help="Fail instead of falling back if the quantum random source is unavailable.",
    )
    generate_parser.add_argument("--seed", type=int, default=7)
    generate_parser.set_defaults(handler=generate)

    return parser.parse_args()


def main():
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
