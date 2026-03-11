import json
import time
import urllib.error
import urllib.request

URL = "http://localhost:8080/v1/embeddings"
HEADERS = {"Content-Type": "application/json"}

TEST_INPUTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "artificial intelligence is transforming the world",
    "machine learning models require large amounts of data",
    "neural networks are inspired by biological brains",
]


def benchmark():
    total_tokenizing = 0.0
    total_inference = 0.0
    total_requests = 0
    total_tokens = 0

    start_time = time.perf_counter()

    print(f"benchmarking {len(TEST_INPUTS)} inputs x 1500 rounds...")
    print("-" * 50)

    for round_num in range(1500):
        for text in TEST_INPUTS:
            payload = json.dumps({"input": text}).encode("utf-8")
            req = urllib.request.Request(
                URL, data=payload, headers=HEADERS, method="POST"
            )

            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8") if e.fp else ""
                print(
                    f"[r{round_num + 1}] '{text[:30]:30s}' HTTP {e.code}: {body[:50]}"
                )
                continue
            except urllib.error.URLError as e:
                print(f"error: {e}")
                return

            elapsed = time.perf_counter() - start

            tokenizing = data.get("tokenizing_time", 0)
            inference = data.get("inference_time", 0)

            total_tokenizing += tokenizing
            total_inference += inference
            total_requests += 1
            total_tokens += data.get("usage", {}).get("total_tokens", 0)

            print(
                f"[r{round_num + 1}] '{text[:30]:30s}' tok:{tokenizing:.4f}ms inf:{inference:.4f}ms"
            )

    print("-" * 50)
    print(f"avg tokenizing: {total_tokenizing / total_requests:.4f}ms")
    print(f"avg inference:   {total_inference / total_requests:.4f}ms")
    print(f"avg tok/prompt: {total_tokens / total_requests:.1f}")

    elapsed_total = time.perf_counter() - start_time
    throughput = total_requests / elapsed_total
    tok_per_sec = total_tokens / elapsed_total

    print("\n" + "=" * 50)

    bar_width = 30
    max_throughput = throughput
    max_tok_per_sec = tok_per_sec

    def make_bar(value, max_val, width):
        filled = int((value / max(max_val, 1)) * width) if max_val > 0 else 0
        return "█" * filled + "░" * (width - filled)

    print(f" throughput: {throughput:7.2f} req/s")
    print(f" {make_bar(throughput, max_throughput, bar_width)}")
    print(f" tok/s:      {tok_per_sec:7.2f}")
    print(f" {make_bar(tok_per_sec, max_tok_per_sec, bar_width)}")
    print(f" total time: {elapsed_total:.2f}s")
    print(f" total reqs: {total_requests}")
    print(f" total tok:  {total_tokens}")


if __name__ == "__main__":
    benchmark()
