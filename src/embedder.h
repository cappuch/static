#ifndef EMBEDDER_H
#define EMBEDDER_H

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer;

class Embedder {
public:
    static constexpr uint32_t EMBEDDING_DIM = 1536;

    Embedder(uint32_t n_vocab, const std::string& embeddings_path = "embeddings.emb");
    ~Embedder();

    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    void load_embeddings(const std::string& path = "");
    void save_binary(const std::string& path = "", uint32_t embedding_dim = EMBEDDING_DIM);
    void load_binary(const std::string& path = "", uint32_t max_token_id = 0);

    std::vector<std::vector<float>> get_token_embeddings(const std::vector<std::string>& texts);
    std::vector<float> get_single_embedding(const std::string& text);

    std::vector<float> get_embedding_from_tokens(const std::vector<uint32_t>& tokens);
    std::vector<std::vector<float>> get_embeddings_from_token_batches(
        const std::vector<std::vector<uint32_t>>& token_batches);

    const int8_t* lookup(uint32_t token_id) const;

    uint32_t embedding_dim() const { return embedding_dim_; }

private:
    uint32_t n_vocab_;
    uint32_t embedding_dim_;
    std::string embeddings_path_;

    std::unordered_map<uint32_t, std::vector<float>> embeddings_dict_;

    int8_t* embeddings_int8_;
    uint32_t flat_capacity_; // max_token_id + 1
    uint8_t* populated_;

    // mmap state
    void* mmap_addr_;
    size_t mmap_len_;

    uint32_t max_token_id_;
    Tokenizer* tokenizer_;
};

#endif
