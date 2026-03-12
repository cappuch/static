# static embedding models
using ideas taken from model2vec, i decided to distil openai's embedding models... this obviously breaks their licensing terms, but who cares?

otherwise, you can *nearly* plug and play with any static model, you just need to convert it to our format.

speed is mindboggling
```
--------------------------------------------------
avg tokenizing: 0.0061ms
avg inference:   0.0021ms
avg tok/prompt: 6.4

==================================================
 throughput: 1472.12 req/s
 ██████████████████████████████
 tok/s:      9421.60
 ██████████████████████████████
 total time: 5.09s
 total reqs: 7500
 total tok:  48000
 ```

performance is eh, might retrain later. (0.41)

## optimizations

obvious ones:
1. distilling an existing model into a static one is instantly faster, no matter the language.
2. using C++ or any like lower-level language will be faster, by default.

c++ backend:
1. int8 values rather than a float. this similifies the memory usage from like ~1gb to ~300mb
2. rather than using a hash map for lookups, allocating a flat contiguous array is faster.
3. all performance-critical buffers use `std::aligned_alloc(64, ...)` or `alignas(64)`. this alighns data to cache line boundaries (64 bytes), which is required for aligned simd load/store insts, and prevents cache line splits
4. accumulation is done with simd
5. if a token appears exactly once, the code skips the multiplication and just does addition.
6. before accumulating, tokens are sorted and dedupes into `(token_id, frequency)` pairs, if a token appears N times, we only look it up once and multiply it by N, rather than looking it up N times.
7. software prefetching hides memory latency by fetching embeddings into L1/L2 cache before they're needed
8. the final int32-to-float conversion also uses simd
9. loading the binary uses mmap with MAP_POPULATE + madvise SEQUENTIAL for fast file I/O
10. deltas between embeddings are stored in the binary, reduces storage and improves quantization precision
11. tcp nodelay is enabled on sockets to send responses immediately without nagle delay
12. http responses use writev (scatter-gather) to send header + body in one syscall
13. json responses are built with a pre-sized buffer + snprintf instead of string streams
14. the accumulator array is stack-allocated to avoid heap allocation per request

that's what makes it so fast


## training
i used a local github copilot proxy for the embeddings, for _free_ embeddings, and just distilled (train/)[train/]
