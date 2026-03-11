#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <vector>
#include <cstdint>

class Embedder;
class Tokenizer;

class Server {
public:
    Server(Embedder* embedder, uint16_t port = 8080);
    ~Server();

    void start();
    void stop();

private:
    Embedder* embedder_;
    Tokenizer* tokenizer_;
    uint16_t port_;
    bool running_;
    int server_fd_;

    void handle_request(int client_fd);
    std::string get_header_value(const std::string& headers, const std::string& key);
    std::string extract_json_body(const std::string& request);
    
    void handle_embeddings(int client_fd, const std::string& body);
    void send_json_response(int client_fd, int status_code, const std::string& json);
    void send_error(int client_fd, const std::string& error);
    std::string get_status_text(int code);
    std::string extract_json_value(const std::string& json, const std::string& key);
    std::vector<std::string> parse_json_array(const std::string& json_array);
};

#endif
