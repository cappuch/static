#ifndef TOKENIZER_WRAPPER_H
#define TOKENIZER_WRAPPER_H

#include <cstdint>
#include <string>
#include <vector>

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

    Tokenizer(Model model = Model::O200K_BASE);
    ~Tokenizer();

    std::vector<uint32_t> encode(const std::string& text);
    std::vector<uint32_t> encode_ordinary(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);

private:
    void* core_bpe_;
    Model model_;
};

class BatchTokenizer {
public:
    BatchTokenizer(Tokenizer::Model model = Tokenizer::Model::O200K_BASE);
    ~BatchTokenizer();

    std::vector<std::vector<uint32_t>> encode(const std::vector<std::string>& texts);

private:
    Tokenizer tokenizer_;
};

#endif
