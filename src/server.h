#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <vector>
#include <cstdint>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

class Embedder;
class Tokenizer;

class Server {
public:
    Server(Embedder* embedder, uint16_t port = 8080, size_t num_threads = 0, const std::string& tokenizer_path = "");
    ~Server();

    void start();
    void stop();

private:
    Embedder* embedder_;
    Tokenizer* tokenizer_;
    uint16_t port_;
    bool running_;
    int server_fd_;

    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::queue<int> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_flag_;

    static constexpr size_t MAX_INPUT_LENGTH = 8192 * 16;
    static constexpr size_t NUM_DEFAULT_THREADS = 4;

    void worker_thread();
    void handle_request(int client_fd, bool& keep_alive);
    std::string get_header_value(const std::string& headers, const std::string& key);
    
    void handle_embeddings(int client_fd, const std::string& body);
    void send_json_response(int client_fd, int status_code, const std::string& json, bool keep_alive);
    void send_error(int client_fd, const std::string& error, bool keep_alive);
    std::string get_status_text(int code);
    
    bool parse_json_body(const std::string& body, std::string& input, std::string& model);
    std::vector<std::string> parse_json_array(const std::string& json_array);
};

#endif
