"""MTEB STS benchmark for the static embedding server.

Wraps the local /v1/embeddings endpoint as an MTEB-compatible model
and evaluates on Semantic Textual Similarity (STS) tasks.

Usage:
    # start the server first, e.g.:  ./build/dist/embedder potion-base-2M.bin
    uv run bench_sts.py
    uv run bench_sts.py --url http://localhost:8080/v1/embeddings --tasks STSBenchmark
    uv run bench_sts.py --tasks all   # run all STS tasks
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Unpack

import mteb
import numpy as np
from mteb.models import ModelMeta

if TYPE_CHECKING:
    from mteb import TaskMetadata
    from mteb.encoder_interface import EncodeKwargs, PromptType
    from mteb.types import Array, BatchedInput, DataLoader

# ── All STS tasks available in MTEB ──────────────────────────────────────────

STS_TASKS = [
    "STSBenchmark",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "BIOSSES",
    "SICK-R",
]

DEFAULT_TASKS = ["STSBenchmark"]


class StaticEmbeddingModel:
    """Wraps the local static HTTP server as an MTEB model."""

    def __init__(self, url: str, model_name: str = "local/static"):
        self.url = url
        self.model_name = model_name
        self._dim: int | None = None

    # ── MTEB interface ───────────────────────────────────────────────────

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return ModelMeta(
            loader=None,
            name=self.model_name,
            revision="local",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=self._dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode sentences by calling the local embedding server."""
        sentences = [text for batch in inputs for text in batch["text"]]

        all_embeddings: list[list[float]] = []
        for text in sentences:
            all_embeddings.append(self._embed_single(text))

        return np.array(all_embeddings, dtype=np.float32)

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Cosine similarity between two sets of embeddings."""
        e1 = np.asarray(embeddings1, dtype=np.float32)
        e2 = np.asarray(embeddings2, dtype=np.float32)
        if e1.ndim == 1:
            e1 = e1[np.newaxis, :]
        if e2.ndim == 1:
            e2 = e2[np.newaxis, :]
        e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True).clip(1e-8)
        e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True).clip(1e-8)
        return e1_norm @ e2_norm.T

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Pairwise cosine similarity between corresponding embeddings."""
        e1 = np.asarray(embeddings1, dtype=np.float32)
        e2 = np.asarray(embeddings2, dtype=np.float32)
        e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True).clip(1e-8)
        e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True).clip(1e-8)
        return np.sum(e1_norm * e2_norm, axis=1)

    # ── internals ────────────────────────────────────────────────────────

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text via the /v1/embeddings endpoint."""
        payload = json.dumps({"input": text}).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code} from embedding server: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach embedding server at {self.url}: {e}"
            ) from e

        embedding = data["data"][0]["embedding"]

        if self._dim is None:
            self._dim = len(embedding)

        return embedding

    def _connectivity_check(self) -> int:
        """Verify the server is reachable, return embedding dim."""
        emb = self._embed_single("connectivity check")
        return len(emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTEB STS benchmarks against the static server"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080/v1/embeddings",
        help="embedding server URL (default: http://localhost:8080/v1/embeddings)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=f"STS tasks to run (default: {DEFAULT_TASKS}). "
        f"Use 'all' to run: {', '.join(STS_TASKS)}",
    )
    parser.add_argument(
        "--output-dir",
        default="mteb_results",
        help="directory for MTEB result files (default: mteb_results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size for encoding (default: 64)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="model name for MTEB metadata / result caching "
        "(default: derived from server URL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_names = STS_TASKS if "all" in args.tasks else args.tasks

    # validate task names
    for t in task_names:
        if t not in STS_TASKS:
            raise SystemExit(
                f"unknown STS task: {t}\navailable: {', '.join(STS_TASKS)}"
            )

    # derive a model name from the URL if not explicitly set
    model_name = args.model_name
    if model_name is None:
        # e.g. "http://localhost:8081/v1/embeddings" -> "local/port-8081"
        from urllib.parse import urlparse

        parsed = urlparse(args.url)
        model_name = f"local/port-{parsed.port or 8080}"
    elif "/" not in model_name:
        model_name = f"local/{model_name}"

    model = StaticEmbeddingModel(url=args.url, model_name=model_name)

    # quick connectivity check
    print(f"server: {args.url}")
    try:
        dim = model._connectivity_check()
        print(f"embedding dim: {dim}")
    except RuntimeError as e:
        raise SystemExit(f"server check failed: {e}") from e

    print(f"tasks: {', '.join(task_names)}")
    print("-" * 60)

    tasks = mteb.get_tasks(tasks=task_names)
    evaluation = mteb.MTEB(tasks=tasks)

    results = evaluation.run(
        model,
        output_folder=args.output_dir,
        batch_size=args.batch_size,
    )

    # ── summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("STS benchmark results")
    print("=" * 60)

    for task_result in results:
        task_name = task_result.task_name
        for split, split_scores in task_result.scores.items():
            for score_entry in split_scores:
                main_score = score_entry.get("main_score", None)
                cosine_spearman = score_entry.get(
                    "cosine_spearman", score_entry.get("cos_sim_spearman", None)
                )
                if main_score is not None and cosine_spearman is not None:
                    print(
                        f"  {task_name:20s} [{split:6s}]  "
                        f"main={main_score:.4f}  "
                        f"spearman={cosine_spearman:.4f}"
                    )
                else:
                    print(f"  {task_name:20s} [{split:6s}]  {score_entry}")

    print("=" * 60)
    print(f"results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
