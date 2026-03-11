import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-3-small")


def tokenize(text: str) -> list[int]:
    return enc.encode(text)


def batch_tokenize(texts: list[str]) -> list[list[int]]:
    return enc.encode_batch(texts)


def return_tokens() -> list[tuple[int, str]]:
    result = []
    for i in range(enc.n_vocab):
        try:
            result.append((i, enc.decode([i])))
        except KeyError:
            pass
    return result


if __name__ == "__main__":
    print(return_tokens())
