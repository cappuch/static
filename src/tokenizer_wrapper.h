#ifndef TOKENIZER_WRAPPER_H
#define TOKENIZER_WRAPPER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <memory>


struct TrieNode {
    static constexpr uint32_t INVALID_ID = UINT32_MAX;
    static constexpr int32_t NO_CHILD = -1;

    uint32_t token_id;                // INVALID_ID if not a terminal
    std::array<int32_t, 256> children; // index into node pool, NO_CHILD if absent

    TrieNode() : token_id(INVALID_ID) {
        children.fill(NO_CHILD);
    }
};

class Tokenizer {
public:
    enum class Model {
        R50K_BASE,
        P50K_BASE,
        P50K_EDIT,
        CL100K_BASE,
        O200K_BASE,
        O200K_HARMONY
    };

    explicit Tokenizer(Model model = Model::O200K_BASE);
    explicit Tokenizer(const std::string& tokenizer_json_path);
    ~Tokenizer();

    std::vector<uint32_t> encode(const std::string& text);
    std::vector<uint32_t> encode_ordinary(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);

    size_t vocab_size() const { return vocab_size_; }

private:
    void* core_bpe_;
    Model model_;
    size_t vocab_size_;

    // Trie-based vocab: separate tries for first-piece and continuation (##) pieces
    std::vector<TrieNode> trie_nodes_;      // pool of trie nodes
    int32_t trie_root_;                      // root for first-piece tokens
    int32_t trie_cont_root_;                 // root for ## continuation tokens
    uint32_t unk_id_;

    std::vector<std::string> id_to_token_;

    bool use_huggingface_;
    std::string tokenizer_json_path_;

    void init_huggingface(const std::string& json_path);
    std::vector<uint32_t> encode_huggingface(const std::string& text);

    int32_t trie_alloc_node();
    void trie_insert(int32_t root, const char* key, size_t len, uint32_t id);
};

class BatchTokenizer {
public:
    explicit BatchTokenizer(Tokenizer::Model model = Tokenizer::Model::O200K_BASE);
    explicit BatchTokenizer(const std::string& tokenizer_json_path);
    ~BatchTokenizer();

    std::vector<std::vector<uint32_t>> encode(const std::vector<std::string>& texts);

private:
    Tokenizer tokenizer_;
};

#endif
