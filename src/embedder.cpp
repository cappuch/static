#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
    , embedding_dim_(EMBEDDING_DIM)
    , embeddings_path_(embeddings_path)
    , embeddings_int8_(nullptr)
    , flat_capacity_(0)
    , populated_(nullptr)
    , mmap_addr_(nullptr)
    , mmap_len_(0)
    , max_token_id_(0)
    , tokenizer_(new Tokenizer(Tokenizer::Model::CL100K_BASE)) {}

Embedder::~Embedder() {
    delete tokenizer_;
    if (embeddings_int8_) {
        std::free(embeddings_int8_);
        embeddings_int8_ = nullptr;
    }
    if (populated_) {
        std::free(populated_);
        populated_ = nullptr;
    }
}

void Embedder::load_embeddings(const std::string& path) {
    std::string filepath = path.empty() ? embeddings_path_ : path;

    std::ifstream file(filepath, std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "cannot open file: " << filepath << std::endl;
        return;
    }
    size_t file_size = file.tellg();
    file.close();

    std::cout << "loading embeddings from " << filepath << " ("
              << (file_size / 1024.0 / 1024.0) << " mb)" << std::endl;

    embeddings_dict_.clear();
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "cannot open file: " << filepath << std::endl;
        return;
    }

    size_t bytes_read = 0;
    std::string line;
    while (std::getline(f, line)) {
        bytes_read += line.size() + 1;
        size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) continue;

        uint32_t token_id = std::stoul(line.substr(0, tab_pos));
        std::string emb_str = line.substr(tab_pos + 1);

        std::vector<float> embedding;
        std::istringstream iss(emb_str);
        float val;
        while (iss >> val) {
            embedding.push_back(val);
        }

        embeddings_dict_[token_id] = std::move(embedding);

        if (embeddings_dict_.size() % 10000 == 0) {
            std::cout << "\rloaded " << embeddings_dict_.size() << " tokens ("
                      << (bytes_read * 100.0 / file_size) << "%)" << std::flush;
        }
    }

    std::cout << "\rloaded " << embeddings_dict_.size() << " tokens ("
              << (file_size / 1024.0 / 1024.0) << " mb)" << std::endl;
}

void Embedder::save_binary(const std::string& path, uint32_t embedding_dim) {
    std::string filepath = path.empty() ?
        embeddings_path_.substr(0, embeddings_path_.find(".emb")) + ".bin" :
        path;

    if (filepath == embeddings_path_) {
        filepath = filepath.replace(filepath.find(".emb"), 4, ".bin");
    }

    BinaryFormat::save(filepath, embeddings_dict_, embedding_dim);
}

void Embedder::load_binary(const std::string& path, uint32_t max_token_id) {
    std::string filepath = path.empty() ? embeddings_path_ : path;

    if (filepath.find(".emb") != std::string::npos) {
        filepath = filepath.replace(filepath.find(".emb"), 4, ".bin");
    }

    std::ifstream test_file(filepath, std::ios::binary);
    if (!test_file.is_open()) {
        std::string fallback = filepath;
        if (fallback.find(".bin") != std::string::npos) {
            fallback = fallback.replace(fallback.find(".bin"), 4, ".emb");
        }
        std::cout << "binary not found, loading text format" << std::endl;
        load_embeddings(fallback);
        return;
    }
    test_file.close();

    max_token_id_ = max_token_id == 0 ? 200000 : max_token_id;
    flat_capacity_ = max_token_id_ + 1;

    if (embeddings_int8_) {
        std::free(embeddings_int8_);
        embeddings_int8_ = nullptr;
    }
    if (populated_) {
        std::free(populated_);
        populated_ = nullptr;
    }

    size_t flat_bytes = static_cast<size_t>(flat_capacity_) * embedding_dim_;
    embeddings_int8_ = static_cast<int8_t*>(std::aligned_alloc(64, flat_bytes));
    if (!embeddings_int8_) {
        std::cerr << "failed to allocate " << (flat_bytes / 1024.0 / 1024.0) << " MB for int8 embeddings" << std::endl;
        return;
    }
    std::memset(embeddings_int8_, 0, flat_bytes);

    size_t pop_bytes = flat_capacity_;
    populated_ = static_cast<uint8_t*>(std::aligned_alloc(64, pop_bytes));
    if (!populated_) {
        std::cerr << "failed to allocate populated array" << std::endl;
        std::free(embeddings_int8_);
        embeddings_int8_ = nullptr;
        return;
    }
    std::memset(populated_, 0, pop_bytes);

    embeddings_dict_.clear();
    uint32_t dim = BinaryFormat::load_flat_int8(filepath, embeddings_int8_, populated_,
                                                 flat_capacity_);
    if (dim > 0) {
        embedding_dim_ = dim;
    }

    std::cout << "int8 flat array: " << (flat_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
}

const int8_t* Embedder::lookup(uint32_t token_id) const {
    if (token_id < flat_capacity_ && populated_[token_id]) {
        return embeddings_int8_ + static_cast<size_t>(token_id) * embedding_dim_;
    }
    return nullptr;
}

static inline void accumulate_scaled(int32_t* __restrict__ sum, const int8_t* __restrict__ emb,
                                     int32_t freq, uint32_t dim) {
// claude is just too sexy
#if defined(HAS_AVX2)
    if (freq == 1) {
        uint32_t j = 0;
        for (; j + 64 <= dim; j += 64) {
            __m256i bytes0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(emb + j));
            __m128i lo0 = _mm256_castsi256_si128(bytes0);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j)),
                                 _mm256_cvtepi8_epi32(lo0)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 8),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 8)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(lo0, 8))));
            __m128i hi0 = _mm256_extracti128_si256(bytes0, 1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 16),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 16)),
                                 _mm256_cvtepi8_epi32(hi0)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 24),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 24)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(hi0, 8))));

            __m256i bytes1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(emb + j + 32));
            __m128i lo1 = _mm256_castsi256_si128(bytes1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 32),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 32)),
                                 _mm256_cvtepi8_epi32(lo1)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 40),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 40)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(lo1, 8))));
            __m128i hi1 = _mm256_extracti128_si256(bytes1, 1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 48),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 48)),
                                 _mm256_cvtepi8_epi32(hi1)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 56),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 56)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(hi1, 8))));
        }
        for (; j + 32 <= dim; j += 32) {
            __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(emb + j));
            __m128i lo = _mm256_castsi256_si128(bytes);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j)),
                                 _mm256_cvtepi8_epi32(lo)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 8),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 8)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))));
            __m128i hi = _mm256_extracti128_si256(bytes, 1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 16),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 16)),
                                 _mm256_cvtepi8_epi32(hi)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 24),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 24)),
                                 _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))));
        }
        for (; j < dim; ++j) sum[j] += emb[j];
    } else {
        __m256i vfreq = _mm256_set1_epi32(freq);
        uint32_t j = 0;
        for (; j + 32 <= dim; j += 32) {
            __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(emb + j));
            __m128i lo = _mm256_castsi256_si128(bytes);
            __m128i hi = _mm256_extracti128_si256(bytes, 1);

            __m256i w0 = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(lo), vfreq);
            __m256i w1 = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)), vfreq);
            __m256i w2 = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(hi), vfreq);
            __m256i w3 = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)), vfreq);

            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j)), w0));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 8),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 8)), w1));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 16),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 16)), w2));
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum + j + 24),
                _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j + 24)), w3));
        }
        for (; j < dim; ++j) sum[j] += emb[j] * freq;
    }
#elif defined(HAS_NEON)
    if (freq == 1) {
        uint32_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            int8x16_t v = vld1q_s8(emb + j);
            int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
            int16x8_t hi16 = vmovl_s8(vget_high_s8(v));
            vst1q_s32(sum + j,      vaddq_s32(vld1q_s32(sum + j),      vmovl_s16(vget_low_s16(lo16))));
            vst1q_s32(sum + j + 4,  vaddq_s32(vld1q_s32(sum + j + 4),  vmovl_s16(vget_high_s16(lo16))));
            vst1q_s32(sum + j + 8,  vaddq_s32(vld1q_s32(sum + j + 8),  vmovl_s16(vget_low_s16(hi16))));
            vst1q_s32(sum + j + 12, vaddq_s32(vld1q_s32(sum + j + 12), vmovl_s16(vget_high_s16(hi16))));
        }
        for (; j < dim; ++j) sum[j] += emb[j];
    } else {
        int32x4_t vfreq = vdupq_n_s32(freq);
        uint32_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            int8x16_t v = vld1q_s8(emb + j);
            int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
            int16x8_t hi16 = vmovl_s8(vget_high_s8(v));
            vst1q_s32(sum + j,      vaddq_s32(vld1q_s32(sum + j),      vmulq_s32(vmovl_s16(vget_low_s16(lo16)), vfreq)));
            vst1q_s32(sum + j + 4,  vaddq_s32(vld1q_s32(sum + j + 4),  vmulq_s32(vmovl_s16(vget_high_s16(lo16)), vfreq)));
            vst1q_s32(sum + j + 8,  vaddq_s32(vld1q_s32(sum + j + 8),  vmulq_s32(vmovl_s16(vget_low_s16(hi16)), vfreq)));
            vst1q_s32(sum + j + 12, vaddq_s32(vld1q_s32(sum + j + 12), vmulq_s32(vmovl_s16(vget_high_s16(hi16)), vfreq)));
        }
        for (; j < dim; ++j) sum[j] += emb[j] * freq;
    }
#else
    for (uint32_t j = 0; j < dim; ++j) sum[j] += emb[j] * freq;
#endif
}

std::vector<float> Embedder::get_embedding_from_tokens(const std::vector<uint32_t>& tokens) {
    if (tokens.empty()) return {};

    if (embeddings_int8_) {
        const uint32_t dim = embedding_dim_;

        // dedupe
        std::vector<uint32_t> sorted_tokens(tokens.begin(), tokens.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end());

        struct TokenFreq { uint32_t id; int32_t freq; };
        std::vector<TokenFreq> unique_tokens;
        unique_tokens.reserve(sorted_tokens.size()); // worst case: all unique
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

        alignas(64) int32_t sum[1536] = {0};
        uint32_t total_count = 0;
        const size_t nu = unique_tokens.size();

        constexpr size_t PREFETCH_DIST = 2;
        constexpr size_t PREFETCH_LINES = 6;

        auto prefetch_embedding = [](const int8_t* emb) {
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
            const int8_t* emb = lookup(unique_tokens[p].id);
            if (emb) prefetch_embedding(emb);
        }

        for (size_t t = 0; t < nu; ++t) {
            const int8_t* emb = lookup(unique_tokens[t].id);
            if (!emb) continue;

            if (t + PREFETCH_DIST < nu) {
                const int8_t* future = lookup(unique_tokens[t + PREFETCH_DIST].id);
                if (future) prefetch_embedding(future);
            }

            int32_t freq = unique_tokens[t].freq;
            accumulate_scaled(sum, emb, freq, dim);
            total_count += freq;
        }

        if (total_count == 0) return {};

        // convert int32 sum to float: result[j] = sum[j] / (total_count * 127.0f)
        const float scale = 1.0f / (static_cast<float>(total_count) * 127.0f);
        std::vector<float> result(dim);

#if defined(HAS_AVX2)
        __m256 vscale = _mm256_set1_ps(scale);
        uint32_t j = 0;
        for (; j + 8 <= dim; j += 8) {
            __m256i si = _mm256_load_si256(reinterpret_cast<const __m256i*>(sum + j));
            _mm256_storeu_ps(result.data() + j, _mm256_mul_ps(_mm256_cvtepi32_ps(si), vscale));
        }
        for (; j < dim; ++j) result[j] = sum[j] * scale;
#elif defined(HAS_NEON)
        float32x4_t vscale = vdupq_n_f32(scale);
        uint32_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            int32x4_t si = vld1q_s32(sum + j);
            vst1q_f32(result.data() + j, vmulq_f32(vcvtq_f32_s32(si), vscale));
        }
        for (; j < dim; ++j) result[j] = sum[j] * scale;
#else
        for (uint32_t j = 0; j < dim; ++j) result[j] = sum[j] * scale;
#endif

        return result;
    }

    const uint32_t dim = embedding_dim_;
    alignas(64) float sum[1536] = {0};
    uint32_t count = 0;

    for (uint32_t token_id : tokens) {
        auto it = embeddings_dict_.find(token_id);
        if (it != embeddings_dict_.end() && !it->second.empty()) {
            const float* emb = it->second.data();
            for (uint32_t j = 0; j < dim; ++j) {
                sum[j] += emb[j];
            }
            ++count;
        }
    }

    if (count == 0) return {};

    const float inv_count = 1.0f / static_cast<float>(count);
    std::vector<float> result(dim);
    for (uint32_t j = 0; j < dim; ++j) {
        result[j] = sum[j] * inv_count;
    }
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
