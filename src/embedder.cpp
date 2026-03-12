#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include "embedder.h"
#include "binary.h"
#include "tokenizer_wrapper.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAS_AVX2 1
#elif defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

Embedder::Embedder(uint32_t n_vocab, const std::string& embeddings_path)
    : n_vocab_(n_vocab)
    , embedding_dim_(DEFAULT_EMBEDDING_DIM)
    , embeddings_path_(embeddings_path)
    , embeddings_int16_(nullptr)
    , flat_capacity_(0)
    , populated_(nullptr)
    , mmap_addr_(nullptr)
    , mmap_len_(0)
    , max_token_id_(0)
    , tokenizer_(new Tokenizer(Tokenizer::Model::CL100K_BASE))
    , owns_tokenizer_(true) {}

Embedder::~Embedder() {
    if (owns_tokenizer_) {
        delete tokenizer_;
    }
    tokenizer_ = nullptr;
    if (embeddings_int16_) {
        std::free(embeddings_int16_);
        embeddings_int16_ = nullptr;
    }
    if (populated_) {
        std::free(populated_);
        populated_ = nullptr;
    }
}

void Embedder::load_embeddings(const std::string& path) {
    std::string filepath = path.empty() ? embeddings_path_ : path;
    std::cerr << "Warning: .emb format is deprecated, using binary format instead" << std::endl;

    std::string binary_path = filepath;
    if (binary_path.find(".emb") != std::string::npos) {
        binary_path = binary_path.replace(binary_path.find(".emb"), 4, ".bin");
    }

    load_binary(binary_path);
}

void Embedder::save_binary(const std::string& path, uint32_t embedding_dim) {
    std::cerr << "Error: save_binary requires .emb format to be loaded first" << std::endl;
    std::cerr << "Please load .emb file first using load_embeddings(), then call save_binary()" << std::endl;
}

void Embedder::load_binary(const std::string& path, uint32_t max_token_id) {
    std::string filepath = path.empty() ? embeddings_path_ : path;

    if (filepath.find(".emb") != std::string::npos) {
        filepath = filepath.replace(filepath.find(".emb"), 4, ".bin");
    }

    std::ifstream test_file(filepath, std::ios::binary);
    if (!test_file.is_open()) {
        std::cerr << "Error: binary file not found: " << filepath << std::endl;
        return;
    }
    test_file.close();

    uint32_t file_dim = BinaryFormat::read_header_dim(filepath);
    if (file_dim > 0) {
        embedding_dim_ = file_dim;
        std::cout << "Embedding dimension: " << embedding_dim_ << std::endl;
    }

    max_token_id_ = max_token_id == 0 ? 200000 : max_token_id;
    flat_capacity_ = max_token_id_ + 1;

    if (embeddings_int16_) {
        std::free(embeddings_int16_);
        embeddings_int16_ = nullptr;
    }
    if (populated_) {
        std::free(populated_);
        populated_ = nullptr;
    }

    size_t flat_bytes = static_cast<size_t>(flat_capacity_) * embedding_dim_ * sizeof(int16_t);
    embeddings_int16_ = static_cast<int16_t*>(std::aligned_alloc(64, flat_bytes));
    if (!embeddings_int16_) {
        std::cerr << "failed to allocate " << (flat_bytes / 1024.0 / 1024.0) << " MB for int16 embeddings" << std::endl;
        return;
    }
    std::memset(embeddings_int16_, 0, flat_bytes);

    size_t pop_bytes = flat_capacity_;
    populated_ = static_cast<uint8_t*>(std::aligned_alloc(64, pop_bytes));
    if (!populated_) {
        std::cerr << "failed to allocate populated array" << std::endl;
        std::free(embeddings_int16_);
        embeddings_int16_ = nullptr;
        return;
    }
    std::memset(populated_, 0, pop_bytes);

    uint32_t dim = BinaryFormat::load_flat_int8(filepath, embeddings_int16_, populated_,
                                                 flat_capacity_);
    if (dim > 0) {
        embedding_dim_ = dim;
    }

    std::cout << "int16 flat array: " << (flat_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
}

const int16_t* Embedder::lookup(uint32_t token_id) const {
    if (token_id < flat_capacity_ && populated_[token_id]) {
        return embeddings_int16_ + static_cast<size_t>(token_id) * embedding_dim_;
    }
    return nullptr;
}

void Embedder::accumulate_scaled(int32_t* sum, const int16_t* emb, int32_t freq, uint32_t dim) {
#if defined(HAS_AVX2)
    if (freq == 1) {
        uint32_t j = 0;
        for (; j + 32 <= dim; j += 32) {
            __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j));
            __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j + 8));
            __m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j + 16));
            __m128i s3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j + 24));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j)),
                                 _mm256_cvtepi16_epi32(s0)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j + 8),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j + 8)),
                                 _mm256_cvtepi16_epi32(s1)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j + 16),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j + 16)),
                                 _mm256_cvtepi16_epi32(s2)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j + 24),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j + 24)),
                                 _mm256_cvtepi16_epi32(s3)));
        }
        for (; j + 8 <= dim; j += 8) {
            __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j)),
                                 _mm256_cvtepi16_epi32(s)));
        }
        for (; j < dim; ++j) sum[j] += emb[j];
    } else {
        __m256i vfreq = _mm256_set1_epi32(freq);
        uint32_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j));
            __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(emb + j + 8));

            __m256i w0 = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(s0), vfreq);
            __m256i w1 = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(s1), vfreq);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j)), w0));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum + j + 8),
                _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j + 8)), w1));
        }
        for (; j < dim; ++j) sum[j] += emb[j] * freq;
    }
#elif defined(HAS_NEON)
    if (freq == 1) {
        uint32_t j = 0;
        for (; j + 8 <= dim; j += 8) {
            int16x8_t v = vld1q_s16(emb + j);
            vst1q_s32(sum + j,     vaddq_s32(vld1q_s32(sum + j),     vmovl_s16(vget_low_s16(v))));
            vst1q_s32(sum + j + 4, vaddq_s32(vld1q_s32(sum + j + 4), vmovl_s16(vget_high_s16(v))));
        }
        for (; j < dim; ++j) sum[j] += emb[j];
    } else {
        int32x4_t vfreq = vdupq_n_s32(freq);
        uint32_t j = 0;
        for (; j + 8 <= dim; j += 8) {
            int16x8_t v = vld1q_s16(emb + j);
            vst1q_s32(sum + j,     vaddq_s32(vld1q_s32(sum + j),     vmulq_s32(vmovl_s16(vget_low_s16(v)), vfreq)));
            vst1q_s32(sum + j + 4, vaddq_s32(vld1q_s32(sum + j + 4), vmulq_s32(vmovl_s16(vget_high_s16(v)), vfreq)));
        }
        for (; j < dim; ++j) sum[j] += emb[j] * freq;
    }
#else
    for (uint32_t j = 0; j < dim; ++j) sum[j] += emb[j] * freq;
#endif
}

void Embedder::convert_to_float(const int32_t* sum, float* result, uint32_t dim, uint32_t total_count) {
    const float scale = 1.0f / (static_cast<float>(total_count) * 127.0f);

#if defined(HAS_AVX2)
    __m256 vscale = _mm256_set1_ps(scale);
    uint32_t j = 0;
    for (; j + 8 <= dim; j += 8) {
        __m256i si = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum + j));
        _mm256_storeu_ps(result + j, _mm256_mul_ps(_mm256_cvtepi32_ps(si), vscale));
    }
    for (; j < dim; ++j) result[j] = sum[j] * scale;
#elif defined(HAS_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    uint32_t j = 0;
    for (; j + 4 <= dim; j += 4) {
        int32x4_t si = vld1q_s32(sum + j);
        vst1q_f32(result + j, vmulq_f32(vcvtq_f32_s32(si), vscale));
    }
    for (; j < dim; ++j) result[j] = sum[j] * scale;
#else
    for (uint32_t j = 0; j < dim; ++j) result[j] = sum[j] * scale;
#endif
}

std::vector<float> Embedder::get_embedding_from_tokens(const std::vector<uint32_t>& tokens) {
    if (tokens.empty()) return {};

    const uint32_t dim = embedding_dim_;

    std::vector<uint32_t> sorted_tokens(tokens.begin(), tokens.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end());

    struct TokenFreq { uint32_t id; int32_t freq; };
    std::vector<TokenFreq> unique_tokens;
    unique_tokens.reserve(sorted_tokens.size());
    {
        size_t i = 0;
        const size_t n = sorted_tokens.size();
        while (i < n) {
            uint32_t tid = sorted_tokens[i];
            int32_t freq = 1;
            while (i + freq < n && sorted_tokens[i + freq] == tid) ++freq;
            unique_tokens.push_back({tid, freq});
            i += freq;
        }
    }

    std::vector<int32_t> sum(dim, 0);
    uint32_t total_count = 0;
    const size_t nu = unique_tokens.size();

    constexpr size_t PREFETCH_DIST = 2;
    constexpr size_t PREFETCH_LINES = 12;

    auto prefetch_embedding = [](const int16_t* emb) {
#if defined(HAS_AVX2)
        for (size_t cl = 0; cl < PREFETCH_LINES; ++cl)
            _mm_prefetch(reinterpret_cast<const char*>(emb + cl * 64), _MM_HINT_T0);
#elif defined(HAS_NEON)
        for (size_t cl = 0; cl < PREFETCH_LINES; ++cl)
            __builtin_prefetch(emb + cl * 64, 0, 3);
#else
        (void)emb;
#endif
    };

    for (size_t p = 0; p < std::min(PREFETCH_DIST, nu); ++p) {
        const int16_t* emb = lookup(unique_tokens[p].id);
        if (emb) prefetch_embedding(emb);
    }

    for (size_t t = 0; t < nu; ++t) {
        const int16_t* emb = lookup(unique_tokens[t].id);
        if (!emb) continue;

        if (t + PREFETCH_DIST < nu) {
            const int16_t* future = lookup(unique_tokens[t + PREFETCH_DIST].id);
            if (future) prefetch_embedding(future);
        }

        int32_t freq = unique_tokens[t].freq;
        accumulate_scaled(sum.data(), emb, freq, dim);
        total_count += freq;
    }

    if (total_count == 0) return {};

    std::vector<float> result(dim);
    convert_to_float(sum.data(), result.data(), dim, total_count);

    return result;
}

std::vector<std::vector<float>> Embedder::get_embeddings_from_token_batches(
    const std::vector<std::vector<uint32_t>>& token_batches)
{
    std::vector<std::vector<float>> results;
    results.reserve(token_batches.size());

    for (const auto& tokens : token_batches) {
        auto emb = get_embedding_from_tokens(tokens);
        if (!emb.empty()) {
            results.push_back(std::move(emb));
        }
    }

    return results;
}

std::vector<std::vector<float>> Embedder::get_token_embeddings(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        std::vector<uint32_t> tokens = tokenizer_->encode_ordinary(text);
        auto emb = get_embedding_from_tokens(tokens);
        if (!emb.empty()) {
            results.push_back(std::move(emb));
        }
    }

    return results;
}

std::vector<float> Embedder::get_single_embedding(const std::string& text) {
    std::vector<uint32_t> tokens = tokenizer_->encode_ordinary(text);
    return get_embedding_from_tokens(tokens);
}
