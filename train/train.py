import asyncio
import os
import struct
import time

from helpers.embed import embed
from helpers.tokenizer import batch_tokenize, return_tokens


class ProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.last_time = time.time()
        self.start_time = time.time()

    def update(self):
        self.current += 1

        if self.current % 100 == 0:
            self.last_time = time.time()
            elapsed = self.last_time - self.start_time
            remaining = (self.total - self.current) * elapsed / self.current
            print(
                f"{self.current}/{self.total} - elapsed: {elapsed:.2f}s - remaining: {remaining:.2f}s",
                end="\r",
            )


class Embedder:
    def __init__(self, n_vocab, embeddings_path="embeddings.emb"):
        self.n_vocab = n_vocab
        self.embeddings_path = embeddings_path
        self.embeddings_dict = {}

    async def train(self, batch_size=256, concurrency=4):
        bar = ProgressBar(self.n_vocab)
        tokens = return_tokens()
        semaphore = asyncio.Semaphore(concurrency)

        with open(self.embeddings_path, "w") as f:

            async def process_batch(batch_tokens):
                async with semaphore:
                    texts = [token[1] for token in batch_tokens]
                    embeddings = await embed(texts)
                    assert embeddings is not None
                    for j, token in enumerate(batch_tokens):
                        f.write(f"{token[0]}\t{' '.join(map(str, embeddings[j]))}\n")
                        bar.update()

            batches = [
                tokens[i : i + batch_size] for i in range(0, len(tokens), batch_size)
            ]
            await asyncio.gather(*[process_batch(b) for b in batches])

    def load_embeddings(self, path=None):
        path = path or self.embeddings_path
        self.embeddings_dict = {}
        file_size = os.path.getsize(path)
        bytes_read = 0
        print(f"loading embeddings from {path} ({file_size / 1024 / 1024:.2f} mb)")
        with open(path, "r") as f:
            for line in f:
                bytes_read += len(line.encode())
                token_id, embedding = line.strip().split("\t")
                self.embeddings_dict[int(token_id)] = list(
                    map(float, embedding.split())
                )
                if len(self.embeddings_dict) % 10000 == 0:
                    print(
                        f"\rloaded {len(self.embeddings_dict)} tokens ({bytes_read / file_size * 100:.1f}%)",
                        end="",
                    )
        print(
            f"\rloaded {len(self.embeddings_dict)} tokens ({file_size / 1024 / 1024:.2f} mb)"
        )

    def save_binary(self, path=None, embedding_dim=1536):
        path = (path or self.embeddings_path).replace(".emb", ".bin")
        print(f"saving binary to {path}...")
        n_vocab = len(self.embeddings_dict)

        sorted_ids = sorted(self.embeddings_dict.keys())

        with open(path, "wb") as f:
            f.write(struct.pack("<III", n_vocab, embedding_dim, 1))  # 1 = delta encoded

            prev_emb = None
            for i, token_id in enumerate(sorted_ids):
                emb = self.embeddings_dict[token_id][:embedding_dim]

                if prev_emb is not None:
                    emb = [emb[j] - prev_emb[j] for j in range(embedding_dim)]

                quantized = [int(round(e * 127)) for e in emb]
                f.write(struct.pack("<I", token_id))
                f.write(struct.pack(f"{embedding_dim}b", *quantized))
                prev_emb = self.embeddings_dict[token_id][:embedding_dim]

                if i % 10000 == 0:
                    print(f"\rSaved {i}/{n_vocab} tokens", end="")

        print(f"\ndone! File size: {os.path.getsize(path) / 1024 / 1024:.2f} mb")

    def load_binary(self, path=None, max_token_id=None):
        path = (path or self.embeddings_path).replace(".emb", ".bin")
        if not os.path.exists(path):
            path = path.replace(".bin", ".emb")
            print("binary not found, loading text format")
            self.load_embeddings(path)
            return

        file_size = os.path.getsize(path)
        print(
            f"loading binary embeddings from {path} ({file_size / 1024 / 1024:.2f} mb)"
        )

        self.embeddings_dict = {}

        with open(path, "rb") as f:
            header = f.read(12)
            n_vocab, embedding_dim, flags = struct.unpack("<III", header)

            if max_token_id is None:
                max_token_id = 200000

            self.embeddings_array = [None] * (max_token_id + 1)

        with open(path, "rb") as f:
            header = f.read(12)
            n_vocab, embedding_dim, flags = struct.unpack("<III", header)

            prev_emb = None
            for i in range(n_vocab):
                header_bytes = f.read(4)
                if not header_bytes:
                    break
                token_id = struct.unpack("<I", header_bytes)[0]
                quantized_bytes = f.read(embedding_dim)
                quantized = list(struct.unpack(f"{embedding_dim}b", quantized_bytes))

                emb = [q / 127.0 for q in quantized]

                if prev_emb is not None:
                    emb = [emb[j] + prev_emb[j] for j in range(embedding_dim)]

                self.embeddings_dict[token_id] = emb
                self.embeddings_array[token_id] = emb
                prev_emb = emb

                if (i + 1) % 10000 == 0:
                    print(f"\rLoaded {i + 1}/{n_vocab} tokens", end="")

        print(
            f"\nloaded {len(self.embeddings_dict)} tokens ({file_size / 1024 / 1024:.2f} MB)"
        )

    def get_token_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        tokens = batch_tokenize(texts)
        all_embeddings = []
        for batch in tokens:
            for token_id in batch:
                if (
                    hasattr(self, "embeddings_array")
                    and self.embeddings_array is not None
                ):
                    if token_id < len(self.embeddings_array):
                        emb = self.embeddings_array[token_id]
                        if emb is not None:
                            all_embeddings.append(emb)
                elif token_id in self.embeddings_dict:
                    all_embeddings.append(self.embeddings_dict[token_id])
            if all_embeddings:
                break
        if not all_embeddings:
            return None
        return [self._mean_pooling([emb]) for emb in all_embeddings]

    def _mean_pooling(self, embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        return [
            sum(emb[i] for emb in embeddings) / len(embeddings)
            for i in range(len(embeddings[0]))
        ]


embedder = Embedder(n_vocab=len(return_tokens()))
# asyncio.run(embedder.train(concurrency=32))

embedder.load_embeddings("embeddings.emb")
embedder.save_binary("embeddings.bin")

embedder.load_binary("embeddings.bin")
"""
print(embedder.get_token_embeddings([""]))
print(embedder.get_token_embeddings(["Testing 1234"]))


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return sum(x**2 for x in a) ** 0.5


def cosine_similarity(a, b):
    return dot_product(a, b) / (norm(a) * norm(b))


test_phrases = ["Testing 1234", "", "Testing 5678"]


async def run_tests():
    for phrase in test_phrases:
        static = embedder.get_token_embeddings([phrase])
        oai = await embed([phrase])
        if static and oai:
            static_emb = static[0]
            oai_emb = oai[0]
            print(
                f"Static: {static_emb[:5]}..., OAI: {oai_emb[:5]}... | Cosine: {cosine_similarity(static_emb, oai_emb)} (higher=more similar)"
            )
        else:
            print(f"Static: {static}, OAI: {oai}")


asyncio.run(run_tests())
"""
