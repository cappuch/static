#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "embedder.h"
#include "binary.h"
#include "tokenizer_wrapper.h"

Embedder::Embedder(uint32_t n_vocab, const std::string& embeddings_path)
    : n_vocab_(n_vocab)
    , embeddings_path_(embeddings_path)
    , max_token_id_(0)
    , tokenizer_(new Tokenizer(Tokenizer::Model::CL100K_BASE)) {}

Embedder::~Embedder() {
    delete tokenizer_;
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
    BinaryFormat::load(filepath, embeddings_dict_, &embeddings_array_, max_token_id_);
}

std::vector<std::vector<float>> Embedder::get_token_embeddings(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        std::vector<uint32_t> tokens = tokenizer_->encode(text);
        if (tokens.empty()) continue;

        const size_t dim = 1536;
        float sum[dim] = {0};
        size_t count = 0;

        for (uint32_t token_id : tokens) {
            const std::vector<float>* emb_ptr = nullptr;

            if (token_id < embeddings_array_.size()) {
                emb_ptr = &embeddings_array_[token_id];
            } else {
                auto it = embeddings_dict_.find(token_id);
                if (it != embeddings_dict_.end()) {
                    emb_ptr = &it->second;
                }
            }

            if (emb_ptr && !emb_ptr->empty()) {
                for (size_t i = 0; i < dim; ++i) {
                    sum[i] += (*emb_ptr)[i];
                }
                ++count;
            }
        }

        if (count > 0) {
            std::vector<float> result(dim);
            for (size_t i = 0; i < dim; ++i) {
                result[i] = sum[i] / static_cast<float>(count);
            }
            results.push_back(std::move(result));
        }
    }

    return results;
}

std::vector<float> Embedder::get_single_embedding(const std::string& text) {
    auto results = get_token_embeddings({text});
    if (results.empty()) {
        return {};
    }
    return results[0];
}
