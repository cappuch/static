#include <iostream>
#include <cstdlib>
#include "embedder.h"
#include "server.h"

int main(int argc, char* argv[]) { // friendship ended with claude big pickle is the goat
    std::string embeddings_path = "train/embeddings.bin";
    uint16_t port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                port = static_cast<uint16_t>(std::atoi(argv[++i]));
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options] [embeddings.bin]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -p, --port PORT    Port to listen on (default: 8080)" << std::endl;
            std::cout << "  -h, --help         Show this help message" << std::endl;
            return 0;
        } else {
            embeddings_path = arg;
        }
    }

    std::cout << "loading embeddings from: " << embeddings_path << std::endl;

    Embedder embedder(200000, embeddings_path);
    embedder.load_binary(embeddings_path);

    std::cout << "starting server on port " << port << std::endl;

    Server server(&embedder, port);
    server.start();

    return 0;
}
