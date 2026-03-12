#!/usr/bin/env python3
"""
Convert static safetensor embedding models to binary format for the C++ server.

Usage:
    python convert.py <model_name> [--output <output_path>] [--token <hf_token>]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install huggingface-hub safetensors")
    sys.exit(1)


def read_safetensor_metadata(sf_path: Path) -> dict:
    with open(sf_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise IOError("Could not read safetensors header length.")
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        metadata_bytes = f.read(header_len)
        if len(metadata_bytes) != header_len:
            raise IOError("Could not read safetensors metadata.")
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    return metadata


def load_f32_tensor_from_safetensors(
    sf_path: Path, tensor_name: str, metadata: dict | None = None
) -> list[list[float]]:
    if metadata is None:
        metadata = read_safetensor_metadata(sf_path)

    tensor_info = metadata.get(tensor_name)
    if not tensor_info:
        raise ValueError(f"Tensor '{tensor_name}' not found in safetensors metadata.")

    dtype = tensor_info.get("dtype", "F32")
    if dtype != "F32":
        raise NotImplementedError(
            f"Only F32 dtype is supported. Found {dtype}. "
            "Support for F16/BF16 would require half-precision float decoding."
        )

    shape = tensor_info["shape"]
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D tensor for embeddings, got shape {shape}")

    num_rows, num_cols = shape
    data_offsets = tensor_info["data_offsets"]
    offset_start, offset_end = data_offsets[0], data_offsets[1]

    tensor_data_size = offset_end - offset_start

    with open(sf_path, "rb") as f:
        header_len_bytes_val = f.read(8)
        header_len_val = struct.unpack("<Q", header_len_bytes_val)[0]
        data_block_start_offset_in_file = 8 + header_len_val

        f.seek(data_block_start_offset_in_file + offset_start)
        tensor_bytes = f.read(tensor_data_size)
        if len(tensor_bytes) != tensor_data_size:
            raise IOError(f"Could not read the full tensor data for '{tensor_name}'.")

    embedding_vectors = []
    current_offset = 0
    for _ in range(num_rows):
        row = []
        for _ in range(num_cols):
            row.append(struct.unpack_from("<f", tensor_bytes, current_offset)[0])
            current_offset += 4
        embedding_vectors.append(row)

    return embedding_vectors


def save_binary(
    embeddings: list[list[float]],
    token_ids: list[int],
    output_path: str,
    embedding_dim: int,
    tokenizer_vocab: dict | None = None,
):
    """Save embeddings to binary format compatible with C++ server."""
    n_vocab = len(embeddings)

    sorted_pairs = sorted(zip(token_ids, embeddings), key=lambda x: x[0])
    sorted_ids = [p[0] for p in sorted_pairs]
    sorted_embs = [p[1] for p in sorted_pairs]

    print(f"Saving binary to {output_path}...")

    with open(output_path, "wb") as f:
        flags = 1  # delta encoded
        f.write(struct.pack("<III", n_vocab, embedding_dim, flags))

        prev_emb = None
        for i, (token_id, emb) in enumerate(zip(sorted_ids, sorted_embs)):
            emb = emb[:embedding_dim]

            if prev_emb is not None:
                delta = [emb[j] - prev_emb[j] for j in range(embedding_dim)]
            else:
                delta = emb[:]

            quantized = [max(-127, min(127, int(round(e * 127)))) for e in delta]

            f.write(struct.pack("<I", token_id))
            f.write(struct.pack(f"{embedding_dim}b", *quantized))
            prev_emb = emb[:]

            if (i + 1) % 10000 == 0:
                print(f"\rSaved {i + 1}/{n_vocab} tokens", end="", flush=True)

    print(f"\nDone! File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    if tokenizer_vocab:
        vocab_path = output_path.replace(".bin", ".vocab.json")
        print(f"Saving tokenizer vocab to {vocab_path}...")
        with open(vocab_path, "w") as f:
            json.dump(tokenizer_vocab, f)
        print(f"Tokenizer vocab saved! ({len(tokenizer_vocab)} tokens)")


def convert_model(
    model_name: str, output_path: str | None = None, token: str | None = None
):
    """Download and convert a model from HuggingFace Hub."""
    cache_dir = "/tmp/huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading model: {model_name}")

    model_path = hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors",
        token=token,
        cache_dir=cache_dir,
    )

    tokenizer_path = hf_hub_download(
        repo_id=model_name, filename="tokenizer.json", token=token, cache_dir=cache_dir
    )

    config_path = None
    try:
        config_path = hf_hub_download(
            repo_id=model_name, filename="config.json", token=token, cache_dir=cache_dir
        )
    except Exception:
        pass

    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")

    print("\nReading safetensor metadata...")
    metadata = read_safetensor_metadata(Path(model_path))

    tensor_name = None
    for key in metadata.keys():
        if key.startswith("__metadata__"):
            continue
        tensor_info = metadata[key]
        if tensor_info.get("dtype") == "F32" and len(tensor_info.get("shape", [])) == 2:
            tensor_name = key
            break

    if tensor_name is None:
        print("Looking for embeddings tensor...")
        for key in metadata.keys():
            if key.startswith("__metadata__"):
                continue
            tensor_info = metadata[key]
            shape = tensor_info.get("shape", [])
            if len(shape) == 2:
                print(f"  Found: {key} - {shape} - {tensor_info.get('dtype')}")
                if tensor_info.get("dtype") in ["F32", "float32"]:
                    tensor_name = key
                    break

    if tensor_name is None:
        tensor_name = "embeddings"

    print(f"\nLoading tensor: {tensor_name}")
    embeddings = load_f32_tensor_from_safetensors(
        Path(model_path), tensor_name, metadata
    )

    num_rows = len(embeddings)
    embedding_dim = len(embeddings[0]) if embeddings else 0
    print(f"Loaded embeddings: {num_rows} x {embedding_dim}")

    from tokenizers import Tokenizer as HFTokenizer

    tokenizer = HFTokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    token_ids = list(range(num_rows))
    if num_rows < vocab_size:
        token_ids = list(range(num_rows))
        print(f"Warning: Embeddings ({num_rows}) < vocab ({vocab_size})")
    elif num_rows > vocab_size:
        token_ids = list(range(vocab_size))
        embeddings = embeddings[:vocab_size]
        print(f"Warning: Embeddings ({num_rows}) > vocab ({vocab_size}), truncating")

    tokenizer_vocab = tokenizer.get_vocab()

    if output_path is None:
        model_short = model_name.replace("/", "_")
        output_path = f"{model_short}.bin"

    save_binary(embeddings, token_ids, output_path, embedding_dim, tokenizer_vocab)

    print(f"\nConversion complete!")
    print(f"Binary file: {output_path}")
    print(f"Run server with: ./build/dist/embedder {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensor models to binary format"
    )
    parser.add_argument(
        "model", help="HuggingFace model name (e.g., minishlab/potion-base-2M)"
    )
    parser.add_argument("--output", "-o", help="Output binary file path")
    parser.add_argument("--token", "-t", help="HuggingFace token for private models")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tensors in model"
    )

    args = parser.parse_args()

    if args.list:
        cache_dir = "/tmp/huggingface_cache"
        model_path = hf_hub_download(
            repo_id=args.model,
            filename="model.safetensors",
            token=args.token,
            cache_dir=cache_dir,
        )
        metadata = read_safetensor_metadata(Path(model_path))
        print(f"\nTensors in {args.model}:")
        for key, info in metadata.items():
            if key.startswith("__metadata__"):
                continue
            shape = info.get("shape", [])
            dtype = info.get("dtype")
            print(f"  {key}: {shape} ({dtype})")
        return

    convert_model(args.model, args.output, args.token)


if __name__ == "__main__":
    main()
