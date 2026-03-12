#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include "embedder.h"
#include "server.h"

int main(int argc, char* argv[]) {
    std::string embeddings_path = "train/embeddings.bin";
    std::string tokenizer_path = "";
    uint16_t port = 8080;
    size_t num_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                port = static_cast<uint16_t>(std::atoi(argv[++i]));
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                num_threads = static_cast<size_t>(std::atoi(argv[++i]));
            }
        } else if (arg == "--tokenizer" || arg == "-tokenizer") {
            if (i + 1 < argc) {
                tokenizer_path = argv[++i];
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options] [embeddings.bin]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -p, --port PORT        Port to listen on (default: 8080)" << std::endl;
            std::cout << "  -t, --threads N        Number of worker threads (default: auto)" << std::endl;
            std::cout << "  --tokenizer PATH       Path to tokenizer.json (HuggingFace format)" << std::endl;
            std::cout << "  -h, --help            Show this help message" << std::endl;
            return 0;
        } else {
            embeddings_path = arg;
        }
    }

    std::cout << "loading embeddings from: " << embeddings_path << std::endl;

    Embedder embedder(200000, embeddings_path);
    embedder.load_binary(embeddings_path);

    if (tokenizer_path.empty()) {
        std::string base = embeddings_path;
        auto dot = base.rfind('.');
        if (dot != std::string::npos) {
            base = base.substr(0, dot);
        }
        std::string candidate = base + ".vocab.json";
        std::ifstream test(candidate);
        if (test.good()) {
            tokenizer_path = candidate;
            std::cout << "auto-detected tokenizer: " << tokenizer_path << std::endl;
        }
    }

    if (!tokenizer_path.empty()) {
        std::cout << "loading tokenizer from: " << tokenizer_path << std::endl;
    }

    std::cout << "starting server on port " << port << std::endl;

    Server server(&embedder, port, num_threads, tokenizer_path);
    server.start();

    return 0;
}
