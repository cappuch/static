#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include "tokenizer_wrapper.h"

extern "C" {
#include "../tiktoken-c/tiktoken.h"
}

Tokenizer::Tokenizer(Model model)
    : core_bpe_(nullptr), model_(model), vocab_size_(0),
      trie_root_(-1), trie_cont_root_(-1), unk_id_(1),
      use_huggingface_(false) {
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
    vocab_size_ = 0;
}

Tokenizer::Tokenizer(const std::string& tokenizer_json_path)
    : core_bpe_(nullptr), model_(Model::CL100K_BASE), vocab_size_(0),
      trie_root_(-1), trie_cont_root_(-1), unk_id_(1),
      use_huggingface_(true), tokenizer_json_path_(tokenizer_json_path) {

    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open tokenizer file: " << tokenizer_json_path << std::endl;
        return;
    }

    init_huggingface(tokenizer_json_path);

    std::cout << "Loaded tokenizer: " << vocab_size_ << " tokens, trie_nodes=" << trie_nodes_.size() << std::endl;
}

int32_t Tokenizer::trie_alloc_node() {
    int32_t idx = static_cast<int32_t>(trie_nodes_.size());
    trie_nodes_.emplace_back();
    return idx;
}

void Tokenizer::trie_insert(int32_t root, const char* key, size_t len, uint32_t id) {
    int32_t node = root;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = static_cast<uint8_t>(key[i]);
        if (trie_nodes_[node].children[c] == TrieNode::NO_CHILD) {
            trie_nodes_[node].children[c] = trie_alloc_node();
        }
        node = trie_nodes_[node].children[c];
    }
    trie_nodes_[node].token_id = id;
}

void Tokenizer::init_huggingface(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();

    std::unordered_map<std::string, uint32_t> token_to_id;

    auto unescape_json_string = [](const std::string& s) -> std::string {
        std::string result;
        result.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                switch (s[i + 1]) {
                    case '"': result += '"'; ++i; break;
                    case '\\': result += '\\'; ++i; break;
                    case '/': result += '/'; ++i; break;
                    case 'n': result += '\n'; ++i; break;
                    case 'r': result += '\r'; ++i; break;
                    case 't': result += '\t'; ++i; break;
                    case 'b': result += '\b'; ++i; break;
                    case 'f': result += '\f'; ++i; break;
                    case 'u': {
                        if (i + 5 < s.size()) {
                            std::string hex = s.substr(i + 2, 4);
                            uint32_t cp = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
                            if (cp >= 0xD800 && cp <= 0xDBFF && i + 11 < s.size()
                                && s[i + 6] == '\\' && s[i + 7] == 'u') {
                                std::string hex2 = s.substr(i + 8, 4);
                                uint32_t cp2 = static_cast<uint32_t>(std::stoul(hex2, nullptr, 16));
                                if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                    i += 6;
                                }
                            }
                            if (cp < 0x80) {
                                result += static_cast<char>(cp);
                            } else if (cp < 0x800) {
                                result += static_cast<char>(0xC0 | (cp >> 6));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            } else if (cp < 0x10000) {
                                result += static_cast<char>(0xE0 | (cp >> 12));
                                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                result += static_cast<char>(0xF0 | (cp >> 18));
                                result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                            i += 5;
                        }
                        break;
                    }
                    default: result += s[i + 1]; ++i; break;
                }
            } else {
                result += s[i];
            }
        }
        return result;
    };

    size_t pos = json_str.find('{');
    if (pos == std::string::npos) return;
    ++pos;

    while (pos < json_str.size()) {
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' ||
               json_str[pos] == '\n' || json_str[pos] == '\r' || json_str[pos] == ',')) {
            ++pos;
        }
        if (pos >= json_str.size() || json_str[pos] == '}') break;

        if (json_str[pos] != '"') break;
        ++pos;
        size_t key_start = pos;
        while (pos < json_str.size()) {
            if (json_str[pos] == '\\' && pos + 1 < json_str.size()) {
                pos += 2;
            } else if (json_str[pos] == '"') {
                break;
            } else {
                ++pos;
            }
        }
        std::string raw_key = json_str.substr(key_start, pos - key_start);
        std::string token = unescape_json_string(raw_key);
        if (pos < json_str.size()) ++pos;

        while (pos < json_str.size() && (json_str[pos] == ':' || json_str[pos] == ' ' || json_str[pos] == '\t')) {
            ++pos;
        }

        size_t val_start = pos;
        while (pos < json_str.size() && json_str[pos] != ',' && json_str[pos] != '}' &&
               json_str[pos] != ' ' && json_str[pos] != '\n') {
            ++pos;
        }
        std::string value_str = json_str.substr(val_start, pos - val_start);

        try {
            uint32_t id = static_cast<uint32_t>(std::stoul(value_str));
            token_to_id[token] = id;

            if (id >= id_to_token_.size()) {
                id_to_token_.resize(id + 1);
            }
            id_to_token_[id] = token;
            vocab_size_ = std::max<uint32_t>(vocab_size_, id + 1);
        } catch (...) {
            // skip invalid entries
        }
    }

    auto unk_it = token_to_id.find("[UNK]");
    if (unk_it != token_to_id.end()) {
        unk_id_ = unk_it->second;
    }

    trie_nodes_.reserve(token_to_id.size() * 8);
    trie_root_ = trie_alloc_node();
    trie_cont_root_ = trie_alloc_node();

    for (const auto& [token, id] : token_to_id) {
        if (token.size() > 2 && token[0] == '#' && token[1] == '#') {
            trie_insert(trie_cont_root_, token.data() + 2, token.size() - 2, id);
        } else {
            trie_insert(trie_root_, token.data(), token.size(), id);
        }
    }

    std::cout << "Loaded HuggingFace tokenizer: " << vocab_size_ << " tokens" << std::endl;
}

Tokenizer::~Tokenizer() {
    if (core_bpe_ != nullptr) {
        tiktoken_destroy_corebpe(static_cast<CoreBPE*>(core_bpe_));
    }
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) {
    return encode_ordinary(text);
}

std::vector<uint32_t> Tokenizer::encode_ordinary(const std::string& text) {
    if (use_huggingface_) {
        return encode_huggingface(text);
    }

    if (core_bpe_ == nullptr) {
        return {};
    }

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

std::vector<uint32_t> Tokenizer::encode_huggingface(const std::string& text) {
    std::vector<uint32_t> result;
    if (text.empty()) return result;

    const size_t len = text.size();
    const char* data = text.data();

    thread_local std::string lower_buf;
    lower_buf.resize(len);
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        lower_buf[i] = (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : static_cast<char>(c);
    }

    const char* ldata = lower_buf.data();
    size_t i = 0;

    while (i < len) {
        unsigned char c = static_cast<unsigned char>(ldata[i]);

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }

        bool is_punct = (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
                        (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);

        if (is_punct) {
            int32_t node = trie_root_;
            int32_t child = trie_nodes_[node].children[c];
            if (child != TrieNode::NO_CHILD && trie_nodes_[child].token_id != TrieNode::INVALID_ID) {
                result.push_back(trie_nodes_[child].token_id);
            } else {
                result.push_back(unk_id_);
            }
            ++i;
            continue;
        }

        size_t word_start = i;
        while (i < len) {
            unsigned char wc = static_cast<unsigned char>(ldata[i]);
            if (wc == ' ' || wc == '\t' || wc == '\n' || wc == '\r') break;
            bool wp = (wc >= 0x21 && wc <= 0x2F) || (wc >= 0x3A && wc <= 0x40) ||
                      (wc >= 0x5B && wc <= 0x60) || (wc >= 0x7B && wc <= 0x7E);
            if (wp) break;
            ++i;
        }
        size_t word_end = i;

        size_t pos = word_start;
        bool is_first = true;
        bool bad = false;
        size_t result_start = result.size(); // bookmark to rollback on failure

        while (pos < word_end) {
            int32_t root = is_first ? trie_root_ : trie_cont_root_;
            int32_t node = root;

            uint32_t best_id = TrieNode::INVALID_ID;
            size_t best_end = pos;
            size_t j = pos;

            while (j < word_end) {
                uint8_t ch = static_cast<uint8_t>(ldata[j]);
                int32_t child = trie_nodes_[node].children[ch];
                if (child == TrieNode::NO_CHILD) break;
                node = child;
                ++j;
                if (trie_nodes_[node].token_id != TrieNode::INVALID_ID) {
                    best_id = trie_nodes_[node].token_id;
                    best_end = j;
                }
            }

            if (best_id == TrieNode::INVALID_ID) {
                bad = true;
                break;
            }

            result.push_back(best_id);
            pos = best_end;
            is_first = false;
        }

        if (bad) {
            result.resize(result_start);
            result.push_back(unk_id_);
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) {
    if (use_huggingface_ && !id_to_token_.empty()) {
        std::string result;
        for (uint32_t id : tokens) {
            if (id < id_to_token_.size()) {
                result += id_to_token_[id];
            }
        }
        return result;
    }

    if (core_bpe_ == nullptr) {
        return "";
    }

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

BatchTokenizer::BatchTokenizer(const std::string& tokenizer_json_path)
    : tokenizer_(tokenizer_json_path) {}

BatchTokenizer::~BatchTokenizer() = default;

std::vector<std::vector<uint32_t>> BatchTokenizer::encode(const std::vector<std::string>& texts) {
    std::vector<std::vector<uint32_t>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        results.push_back(tokenizer_.encode_ordinary(text));
    }

    return results;
}
