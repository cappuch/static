#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "binary.h"

namespace {

size_t get_file_size(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return 0;
    return f.tellg();
}

}

void BinaryFormat::save(
    const std::string& filepath,
    const std::unordered_map<uint32_t, std::vector<float>>& embeddings_dict,
    uint32_t embedding_dim
) {
    std::cout << "saving binary to " << filepath << "..." << std::endl;

    uint32_t n_vocab = static_cast<uint32_t>(embeddings_dict.size());

    std::vector<uint32_t> sorted_ids;
    sorted_ids.reserve(embeddings_dict.size());
    for (const auto& kv : embeddings_dict) {
        sorted_ids.push_back(kv.first);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open file for writing: " << filepath << std::endl;
        return;
    }

    uint32_t flags = DELTA_ENCODED;
    f.write(reinterpret_cast<const char*>(&n_vocab), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));

    std::vector<float> prev_emb;
    for (size_t i = 0; i < sorted_ids.size(); ++i) {
        uint32_t token_id = sorted_ids[i];
        const std::vector<float>& emb = embeddings_dict.at(token_id);

        std::vector<float> delta_emb(embedding_dim);
        if (!prev_emb.empty()) {
            for (uint32_t j = 0; j < embedding_dim; ++j) {
                delta_emb[j] = emb[j] - prev_emb[j];
            }
        } else {
            for (uint32_t j = 0; j < embedding_dim; ++j) {
                delta_emb[j] = emb[j];
            }
        }

        int8_t quantized[1536];
        for (uint32_t j = 0; j < embedding_dim; ++j) {
            quantized[j] = static_cast<int8_t>(std::round(delta_emb[j] * 127));
        }

        f.write(reinterpret_cast<const char*>(&token_id), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(quantized), embedding_dim);

        prev_emb = emb;

        if (i % 10000 == 0) {
            std::cout << "\rsaved " << i << "/" << n_vocab << " tokens" << std::flush;
        }
    }

    f.close();
    std::cout << "\ndone! File size: " << (get_file_size(filepath) / 1024.0 / 1024.0) << " mb" << std::endl;
}

void BinaryFormat::load(
    const std::string& filepath,
    std::unordered_map<uint32_t, std::vector<float>>& embeddings_dict,
    std::vector<std::vector<float>>* embeddings_array,
    uint32_t max_token_id
) {
    std::ifstream test_file(filepath, std::ios::binary);
    if (!test_file.is_open()) {
        std::cerr << "binary file not found: " << filepath << std::endl;
        return;
    }
    test_file.close();

    size_t file_size = get_file_size(filepath);
    std::cout << "loading binary embeddings from " << filepath << " ("
              << (file_size / 1024.0 / 1024.0) << " mb)" << std::endl;

    embeddings_dict.clear();

    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "cannot open file: " << filepath << std::endl;
        return;
    }

    uint32_t n_vocab_read, embedding_dim, flags;
    f.read(reinterpret_cast<char*>(&n_vocab_read), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&embedding_dim), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&flags), sizeof(uint32_t));

    if (embeddings_array != nullptr) {
        embeddings_array->clear();
        embeddings_array->resize(max_token_id + 1);
    }

    std::vector<float> prev_emb;
    for (uint32_t i = 0; i < n_vocab_read; ++i) {
        uint32_t token_id;
        if (!f.read(reinterpret_cast<char*>(&token_id), sizeof(uint32_t))) {
            break;
        }

        int8_t quantized[1536];
        if (!f.read(reinterpret_cast<char*>(quantized), embedding_dim)) {
            break;
        }

        std::vector<float> emb(embedding_dim);
        for (uint32_t j = 0; j < embedding_dim; ++j) {
            emb[j] = quantized[j] / 127.0f;
        }

        if (!prev_emb.empty()) {
            for (uint32_t j = 0; j < embedding_dim; ++j) {
                emb[j] += prev_emb[j];
            }
        }

        embeddings_dict[token_id] = emb;
        if (embeddings_array != nullptr && token_id <= max_token_id) {
            (*embeddings_array)[token_id] = emb;
        }
        prev_emb = emb;

        if ((i + 1) % 10000 == 0) {
            std::cout << "\rloaded " << (i + 1) << "/" << n_vocab_read << " tokens" << std::flush;
        }
    }

    f.close();
    std::cout << "\nloaded " << embeddings_dict.size() << " tokens ("
              << (file_size / 1024.0 / 1024.0) << " MB)" << std::endl;
}

size_t BinaryFormat::get_file_size(const std::string& filepath) {
    return ::get_file_size(filepath);
}
