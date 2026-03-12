#ifndef EMBEDDER_H
#define EMBEDDER_H

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <optional>
#include <expected>

class Tokenizer;

enum class EmbedderError {
    FileNotFound,
    InvalidFormat,
    AllocationFailed,
    TokenNotFound
};

class Embedder {
public:
    static constexpr uint32_t DEFAULT_EMBEDDING_DIM = 1536;

    explicit Embedder(uint32_t n_vocab = 200000, const std::string& embeddings_path = "embeddings.emb");
    ~Embedder();

    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    void load_embeddings(const std::string& path = "");
    void save_binary(const std::string& path = "", uint32_t embedding_dim = DEFAULT_EMBEDDING_DIM);
    void load_binary(const std::string& path = "", uint32_t max_token_id = 0);

    std::vector<std::vector<float>> get_token_embeddings(const std::vector<std::string>& texts);
    std::vector<float> get_single_embedding(const std::string& text);

    std::vector<float> get_embedding_from_tokens(const std::vector<uint32_t>& tokens);
    std::vector<std::vector<float>> get_embeddings_from_token_batches(
        const std::vector<std::vector<uint32_t>>& token_batches);

    const int16_t* lookup(uint32_t token_id) const;

    uint32_t embedding_dim() const { return embedding_dim_; }

    void set_tokenizer(Tokenizer* tokenizer) { 
        owns_tokenizer_ = false;
        tokenizer_ = tokenizer; 
    }

private:
    uint32_t n_vocab_;
    uint32_t embedding_dim_;
    std::string embeddings_path_;

    int16_t* embeddings_int16_;
    uint32_t flat_capacity_;
    uint8_t* populated_;

    void* mmap_addr_;
    size_t mmap_len_;

    uint32_t max_token_id_;
    Tokenizer* tokenizer_;
    bool owns_tokenizer_;

    void accumulate_scaled(int32_t* sum, const int16_t* emb, int32_t freq, uint32_t dim);
    void convert_to_float(const int32_t* sum, float* result, uint32_t dim, uint32_t total_count);
};

#endif
