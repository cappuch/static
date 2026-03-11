import asyncio

import httpx


async def embed_with_retry(
    texts: list[str], max_retries: int = 5
) -> list[list[float]] | None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    "http://localhost:4141/v1/embeddings",
                    json={"input": texts, "model": "text-embedding-3-small"},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}, retrying in {wait}s..."
                )
                await asyncio.sleep(wait)


async def embed(texts: list[str]) -> list[list[float]] | None:
    return await embed_with_retry(texts)


if __name__ == "__main__":
    print(asyncio.run(embed(["test", ""])))
