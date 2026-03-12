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
    : core_bpe_(nullptr), model_(model), vocab_size_(0), max_token_len_(0), use_huggingface_(false) {
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
    : core_bpe_(nullptr), model_(Model::CL100K_BASE), vocab_size_(0), max_token_len_(0), use_huggingface_(true),
      tokenizer_json_path_(tokenizer_json_path) {

    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open tokenizer file: " << tokenizer_json_path << std::endl;
        return;
    }

    init_huggingface(tokenizer_json_path);

    std::cout << "Loaded tokenizer: " << vocab_size_ << " tokens, max_token_len=" << max_token_len_ << std::endl;
}

void Tokenizer::init_huggingface(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();

    max_token_len_ = 0;

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
                                    i += 6; // skip second \uXXXX
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
        if (pos < json_str.size()) ++pos; // skip closing quote

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
            token_to_id_[token] = id;

            if (id >= id_to_token_.size()) {
                id_to_token_.resize(id + 1);
            }
            id_to_token_[id] = token;
            vocab_size_ = std::max<uint32_t>(vocab_size_, id + 1);

            size_t effective_len = token.size();
            if (token.size() > 2 && token[0] == '#' && token[1] == '#') {
                effective_len = token.size() - 2;
            }
            if (effective_len > max_token_len_) {
                max_token_len_ = effective_len;
            }
        } catch (...) {
            // skip invalid entries
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

    std::string lower_text;
    lower_text.reserve(text.size());
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z') {
            lower_text += static_cast<char>(c + 32);
        } else {
            lower_text += static_cast<char>(c);
        }
    }

    std::vector<std::string> words;
    std::string current_word;
    for (size_t i = 0; i < lower_text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(lower_text[i]);
        bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        bool is_punct = (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
                        (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);

        if (is_space) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
        } else if (is_punct) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            words.push_back(std::string(1, static_cast<char>(c)));
        } else {
            current_word += static_cast<char>(c);
        }
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    uint32_t unk_id = 1; // usually is the unk id
    auto unk_it = token_to_id_.find("[UNK]");
    if (unk_it != token_to_id_.end()) {
        unk_id = unk_it->second;
    }

    for (const auto& word : words) {
        size_t start = 0;
        bool bad = false;
        std::vector<uint32_t> word_tokens;

        while (start < word.size()) {
            size_t end = word.size();
            bool found = false;

            if (start == 0) {
                if (end - start > max_token_len_) {
                    end = start + max_token_len_;
                }
            } else {
                if (end - start > max_token_len_) {
                    end = start + max_token_len_;
                }
            }

            while (end > start) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }

                auto it = token_to_id_.find(substr);
                if (it != token_to_id_.end()) {
                    word_tokens.push_back(it->second);
                    start = end;
                    found = true;
                    break;
                }

                --end;
                while (end > start && (static_cast<unsigned char>(word[end]) & 0xC0) == 0x80) {
                    --end;
                }
            }

            if (!found) {
                bad = true;
                break;
            }
        }

        if (bad) {
            result.push_back(unk_id);
        } else {
            result.insert(result.end(), word_tokens.begin(), word_tokens.end());
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
