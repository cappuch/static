#ifndef EMBEDDER_H
#define EMBEDDER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer;

class Embedder {
public:
    Embedder(uint32_t n_vocab, const std::string& embeddings_path = "embeddings.emb");
    ~Embedder();

    void load_embeddings(const std::string& path = "");
    void save_binary(const std::string& path = "", uint32_t embedding_dim = 1536);
    void load_binary(const std::string& path = "", uint32_t max_token_id = 0);

    std::vector<std::vector<float>> get_token_embeddings(const std::vector<std::string>& texts);
    std::vector<float> get_single_embedding(const std::string& text);

private:
    uint32_t n_vocab_;
    std::string embeddings_path_;
    std::unordered_map<uint32_t, std::vector<float>> embeddings_dict_;
    std::vector<std::vector<float>> embeddings_array_;
    uint32_t max_token_id_;
    Tokenizer* tokenizer_;
};

#endif
