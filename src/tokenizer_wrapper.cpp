#include <string>
#include <vector>
#include "tokenizer_wrapper.h"

extern "C" {
#include "../tiktoken-c/tiktoken.h"
}

Tokenizer::Tokenizer(Model model) : model_(model) {
    switch (model) {
        case Model::R50K_BASE:
            core_bpe_ = tiktoken_r50k_base();
            break;
        case Model::P50K_BASE:
            core_bpe_ = tiktoken_p50k_base();
            break;
        case Model::P50K_EDIT:
            core_bpe_ = tiktoken_p50k_edit();
            break;
        case Model::CL100K_BASE:
            core_bpe_ = tiktoken_cl100k_base();
            break;
        case Model::O200K_BASE:
            core_bpe_ = tiktoken_o200k_base();
            break;
        case Model::O200K_HARMONY:
            core_bpe_ = tiktoken_o200k_harmony();
            break;
    }
}

Tokenizer::~Tokenizer() {
    if (core_bpe_ != nullptr) {
        tiktoken_destroy_corebpe(static_cast<CoreBPE*>(core_bpe_));
    }
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) {
    size_t num_tokens = 0;
    CoreBPE* bpe = static_cast<CoreBPE*>(core_bpe_);
    Rank* tokens = tiktoken_corebpe_encode_with_special_tokens(bpe, text.c_str(), &num_tokens);

    std::vector<uint32_t> result(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        result[i] = static_cast<uint32_t>(tokens[i]);
    }

    tiktoken_free(tokens);
    return result;
}

std::vector<uint32_t> Tokenizer::encode_ordinary(const std::string& text) {
    size_t num_tokens = 0;
    CoreBPE* bpe = static_cast<CoreBPE*>(core_bpe_);
    Rank* tokens = tiktoken_corebpe_encode_ordinary(bpe, text.c_str(), &num_tokens);

    std::vector<uint32_t> result(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        result[i] = static_cast<uint32_t>(tokens[i]);
    }

    tiktoken_free(tokens);
    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) {
    CoreBPE* bpe = static_cast<CoreBPE*>(core_bpe_);
    Rank* ranks = new Rank[tokens.size()];
    for (size_t i = 0; i < tokens.size(); ++i) {
        ranks[i] = static_cast<Rank>(tokens[i]);
    }

    char* decoded = tiktoken_corebpe_decode(bpe, ranks, tokens.size());
    std::string result(decoded);

    tiktoken_free(decoded);
    delete[] ranks;

    return result;
}

BatchTokenizer::BatchTokenizer(Tokenizer::Model model)
    : tokenizer_(model) {}

BatchTokenizer::~BatchTokenizer() = default;

std::vector<std::vector<uint32_t>> BatchTokenizer::encode(const std::vector<std::string>& texts) {
    std::vector<std::vector<uint32_t>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        results.push_back(tokenizer_.encode(text));
    }

    return results;
}
