#ifndef BINARY_H
#define BINARY_H

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class BinaryFormat {
public:
    static constexpr uint32_t DELTA_ENCODED = 1;

    static void save(
        const std::string& filepath,
        const std::unordered_map<uint32_t, std::vector<float>>& embeddings_dict,
        uint32_t embedding_dim = 1536
    );

    static void load(
        const std::string& filepath,
        std::unordered_map<uint32_t, std::vector<float>>& embeddings_dict,
        std::vector<std::vector<float>>* embeddings_array = nullptr,
        uint32_t max_token_id = 200000
    );

    static uint32_t read_header_dim(const std::string& filepath);
    
    static uint32_t load_flat_int8(
        const std::string& filepath,
        int16_t* embeddings_int16,
        uint8_t* populated,
        uint32_t flat_capacity
    );

    static size_t get_file_size(const std::string& filepath);
};

#endif
