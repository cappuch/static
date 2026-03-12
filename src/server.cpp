#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/uio.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <atomic>

#include "server.h"
#include "embedder.h"
#include "tokenizer_wrapper.h"

Server::Server(Embedder* embedder, uint16_t port, size_t num_threads, const std::string& tokenizer_path)
    : embedder_(embedder)
    , tokenizer_(nullptr)
    , port_(port)
    , running_(false)
    , server_fd_(-1)
    , num_threads_(num_threads == 0 ? NUM_DEFAULT_THREADS : num_threads)
    , stop_flag_(false)
{
    if (!tokenizer_path.empty()) {
        tokenizer_ = new Tokenizer(tokenizer_path);
    } else {
        tokenizer_ = new Tokenizer(Tokenizer::Model::CL100K_BASE);
    }
    embedder_->set_tokenizer(tokenizer_);
}

Server::~Server() {
    delete tokenizer_;
    stop();
}

void Server::start() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind to port " << port_ << std::endl;
        close(server_fd_);
        return;
    }

    if (listen(server_fd_, 128) < 0) {
        std::cerr << "Failed to listen" << std::endl;
        close(server_fd_);
        return;
    }

    running_ = true;
    stop_flag_.store(false);

    size_t threads_to_use = num_threads_;
    if (threads_to_use == 0) {
        threads_to_use = std::max(1u, std::thread::hardware_concurrency() - 1);
    }

    for (size_t i = 0; i < threads_to_use; ++i) {
        workers_.emplace_back(&Server::worker_thread, this);
    }

    std::cout << "Server started on http://0.0.0.0:" << port_ << std::endl;
    std::cout << "OpenAI-compatible endpoint: http://0.0.0.0:" << port_ << "/v1/embeddings" << std::endl;
    std::cout << "Worker threads: " << threads_to_use << std::endl;

    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            if (running_) {
                std::cerr << "Failed to accept connection" << std::endl;
            }
            continue;
        }

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &flag, sizeof(flag));

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            work_queue_.push(client_fd);
        }
        queue_cv_.notify_one();
    }
}

void Server::stop() {
    running_ = false;
    stop_flag_.store(true);
    queue_cv_.notify_all();

    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

void Server::worker_thread() {
    while (!stop_flag_.load()) {
        int client_fd = -1;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return stop_flag_.load() || !work_queue_.empty();
            });

            if (stop_flag_.load() && work_queue_.empty()) {
                break;
            }

            if (!work_queue_.empty()) {
                client_fd = work_queue_.front();
                work_queue_.pop();
            }
        }

        if (client_fd >= 0) {
            bool keep_alive = true;
            while (keep_alive && !stop_flag_.load()) {
                handle_request(client_fd, keep_alive);

                char peek_buf[1];
                int peek_result = recv(client_fd, peek_buf, 1, MSG_PEEK);
                if (peek_result <= 0) {
                    keep_alive = false;
                }
            }
            close(client_fd);
        }
    }
}

void Server::handle_request(int client_fd, bool& keep_alive) {
    char buffer[16384] = {0};
    std::string request;
    request.reserve(4096);
    ssize_t bytes_read;

    auto set_timeout = [](int fd, int seconds) {
        struct timeval tv;
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    };

    set_timeout(client_fd, keep_alive ? 30 : 5);

    while ((bytes_read = read(client_fd, buffer, sizeof(buffer) - 1)) > 0) {
        request.append(buffer, bytes_read);

        size_t header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            std::string headers = request.substr(0, header_end);
            std::string content_length_str = get_header_value(headers, "Content-Length:");

            size_t body_start = header_end + 4;
            size_t content_length = 0;

            if (!content_length_str.empty()) {
                try {
                    content_length = std::stoul(content_length_str);
                } catch (...) {
                    content_length = 0;
                }
            }

            if (content_length == 0 || body_start + content_length <= request.size()) {
                break;
            }
        }

        if (request.size() > MAX_INPUT_LENGTH * 2) {
            send_error(client_fd, "Request too large", false);
            return;
        }
    }

    set_timeout(client_fd, 0);

    if (request.empty()) {
        return;
    }

    std::istringstream request_stream(request);
    std::string method, path, version;
    request_stream >> method >> path >> version;

    if (method.empty() || path.empty()) {
        send_error(client_fd, "Malformed request", keep_alive);
        return;
    }

    std::string headers;
    std::string line;
    std::getline(request_stream, line);
    while (std::getline(request_stream, line) && line != "\r" && line != "") {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        headers += line + "\n";
    }

    std::string connection = get_header_value(headers, "Connection:");
    bool request_keep_alive = (connection == "keep-alive" || version == "HTTP/1.1");
    if (connection == "close") {
        request_keep_alive = false;
    }
    keep_alive = request_keep_alive;

    std::string body;
    std::string content_length_str = get_header_value(headers, "Content-Length:");
    if (!content_length_str.empty()) {
        try {
            size_t content_length = std::stoul(content_length_str);
            size_t header_end = request.find("\r\n\r\n");
            if (header_end != std::string::npos && header_end + 4 + content_length <= request.size()) {
                body = request.substr(header_end + 4, content_length);
            }
        } catch (...) {
            body = "";
        }
    }

    if (body.empty() && method == "POST") {
        send_error(client_fd, "empty body", request_keep_alive);
        return;
    }

    if (path == "/v1/embeddings" && method == "POST") {
        handle_embeddings(client_fd, body);
    } else if (path == "/health" && method == "GET") {
        send_json_response(client_fd, 200, "{\"status\":\"ok\"}", request_keep_alive);
    } else if (path == "/v1/models" && method == "GET") {
        send_json_response(client_fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"text-embedding-ada-002\",\"object\":\"model\",\"owned_by\":\"openai\",\"permission\":[]}]}", request_keep_alive);
    } else {
        send_error(client_fd, "Not Found", request_keep_alive);
    }
}

std::string Server::get_header_value(const std::string& headers, const std::string& key) {
    std::istringstream stream(headers);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.length() >= key.length() &&
            std::equal(key.begin(), key.end(), line.begin())) {
            std::string value = line.substr(key.length());
            size_t start = value.find_first_not_of(" \t");
            if (start != std::string::npos) {
                value = value.substr(start);
            }
            size_t end = value.find_last_of(" \r");
            if (end != std::string::npos) {
                value = value.substr(0, end);
            }
            return value;
        }
    }
    return "";
}

void Server::handle_embeddings(int client_fd, const std::string& body) {
    if (body.empty()) {
        send_error(client_fd, "Invalid request body", true);
        return;
    }

    if (body.size() > MAX_INPUT_LENGTH) {
        send_error(client_fd, "Input too large", true);
        return;
    }

    std::string input;
    std::string model;

    if (!parse_json_body(body, input, model)) {
        send_error(client_fd, "Invalid JSON body", true);
        return;
    }

    if (input.empty()) {
        send_error(client_fd, "Missing 'input' field", true);
        return;
    }

    if (input.size() > MAX_INPUT_LENGTH) {
        send_error(client_fd, "Input exceeds maximum length", true);
        return;
    }

    std::vector<std::string> texts;
    if (input[0] == '[') {
        texts = parse_json_array(input);
    } else {
        texts = {input};
    }

    if (texts.empty()) {
        send_error(client_fd, "Invalid 'input' field", true);
        return;
    }

    auto start_tokenize = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint32_t>> all_tokens;
    all_tokens.reserve(texts.size());
    size_t total_tokens = 0;
    for (const auto& text : texts) {
        if (text.size() > MAX_INPUT_LENGTH) {
            send_error(client_fd, "Text in array exceeds maximum length", true);
            return;
        }
        all_tokens.push_back(tokenizer_->encode_ordinary(text));
        total_tokens += all_tokens.back().size();
    }
    auto end_tokenize = std::chrono::high_resolution_clock::now();

    auto start_inference = std::chrono::high_resolution_clock::now();
    auto embeddings = embedder_->get_embeddings_from_token_batches(all_tokens);
    auto end_inference = std::chrono::high_resolution_clock::now();

    double tokenizing_time = std::chrono::duration<double, std::milli>(end_tokenize - start_tokenize).count();
    double inference_time = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();

    const char* model_str = model.empty() ? "text-embedding-ada-002" : model.c_str();

    size_t dim = embedder_->embedding_dim();
    size_t est_size = 512 + embeddings.size() * (dim * 16 + 512);
    std::string json_buf;
    json_buf.reserve(est_size);

    json_buf += "{\"object\":\"list\",\"data\":[";

    char float_buf[32];
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (i > 0) json_buf += ',';
        json_buf += "{\"object\":\"embedding\",\"embedding\":[";

        const auto& emb = embeddings[i];
        for (size_t j = 0; j < emb.size(); ++j) {
            if (j > 0) json_buf += ',';
            int len = snprintf(float_buf, sizeof(float_buf), "%.8g", emb[j]);
            json_buf.append(float_buf, len);
        }

        json_buf += "],\"index\":";
        json_buf += std::to_string(i);
        json_buf += ",\"model\":\"";
        json_buf += model_str;
        json_buf += "\"}";
    }

    json_buf += "],\"model\":\"";
    json_buf += model_str;
    json_buf += "\",\"usage\":{\"prompt_tokens\":";
    json_buf += std::to_string(total_tokens);
    json_buf += ",\"total_tokens\":";
    json_buf += std::to_string(total_tokens);
    json_buf += "},\"tokenizing_time\":";
    snprintf(float_buf, sizeof(float_buf), "%.3f", tokenizing_time);
    json_buf += float_buf;
    json_buf += ",\"inference_time\":";
    snprintf(float_buf, sizeof(float_buf), "%.3f", inference_time);
    json_buf += float_buf;
    json_buf += '}';

    send_json_response(client_fd, 200, json_buf, true);
}

bool Server::parse_json_body(const std::string& json, std::string& input, std::string& model) {
    input.clear();
    model.clear();

    auto extract_value = [](const std::string& j, const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        size_t key_pos = j.find(search);
        if (key_pos == std::string::npos) return "";

        size_t colon_pos = j.find(":", key_pos);
        if (colon_pos == std::string::npos) return "";

        size_t value_start = colon_pos + 1;
        while (value_start < j.size() && (j[value_start] == ' ' || j[value_start] == '\t')) {
            value_start++;
        }

        if (value_start >= j.size()) return "";

        if (j[value_start] == '[') {
            int depth = 0;
            size_t value_end = value_start;
            while (value_end < j.size()) {
                if (j[value_end] == '[') depth++;
                else if (j[value_end] == ']') { depth--; if (depth == 0) { value_end++; break; } }
                else if (j[value_end] == '"') {
                    value_end++;
                    while (value_end < j.size() && j[value_end] != '"') {
                        if (j[value_end] == '\\') value_end++;
                        value_end++;
                    }
                }
                value_end++;
            }
            return j.substr(value_start, value_end - value_start);
        }

        if (j[value_start] == '"') {
            value_start++;
            size_t value_end = value_start;
            while (value_end < j.size() && j[value_end] != '"') {
                if (j[value_end] == '\\' && value_end + 1 < j.size()) {
                    value_end += 2;
                } else {
                    value_end++;
                }
            }
            std::string result = j.substr(value_start, value_end - value_start);

            std::string unescaped;
            unescaped.reserve(result.size());
            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i] == '\\' && i + 1 < result.size()) {
                    switch (result[i + 1]) {
                        case 'n': unescaped += '\n'; break;
                        case 'r': unescaped += '\r'; break;
                        case 't': unescaped += '\t'; break;
                        case '"': unescaped += '"'; break;
                        case '\\': unescaped += '\\'; break;
                        default: unescaped += result[i + 1]; break;
                    }
                    i++;
                } else {
                    unescaped += result[i];
                }
            }
            return unescaped;
        }

        size_t value_end = value_start;
        while (value_end < j.size() && j[value_end] != ',' && j[value_end] != '}' && j[value_end] != ']' && j[value_end] != '\n') {
            value_end++;
        }
        return j.substr(value_start, value_end - value_start);
    };

    input = extract_value(json, "input");
    model = extract_value(json, "model");

    return true;
}

std::vector<std::string> Server::parse_json_array(const std::string& json_array) {
    std::vector<std::string> result;
    size_t pos = 0;

    while (pos < json_array.size()) {
        if (json_array[pos] == '"') {
            pos++;
            size_t start = pos;
            while (pos < json_array.size()) {
                if (json_array[pos] == '\\' && pos + 1 < json_array.size()) {
                    pos += 2;
                } else if (json_array[pos] == '"') {
                    break;
                } else {
                    pos++;
                }
            }

            std::string elem = json_array.substr(start, pos - start);

            std::string unescaped;
            unescaped.reserve(elem.size());
            for (size_t i = 0; i < elem.size(); ++i) {
                if (elem[i] == '\\' && i + 1 < elem.size()) {
                    switch (elem[i + 1]) {
                        case 'n': unescaped += '\n'; break;
                        case 'r': unescaped += '\r'; break;
                        case 't': unescaped += '\t'; break;
                        case '"': unescaped += '"'; break;
                        case '\\': unescaped += '\\'; break;
                        default: unescaped += elem[i + 1]; break;
                    }
                    i++;
                } else {
                    unescaped += elem[i];
                }
            }
            result.push_back(unescaped);

            if (pos < json_array.size() && json_array[pos] == '"') pos++;
        } else if (json_array[pos] == ' ' || json_array[pos] == '\t' ||
                   json_array[pos] == '\n' || json_array[pos] == '\r' ||
                   json_array[pos] == ',' || json_array[pos] == '[') {
            pos++;
        } else if (json_array[pos] == ']') {
            break;
        } else {
            size_t start = pos;
            while (pos < json_array.size() && json_array[pos] != ',' && json_array[pos] != ']') {
                pos++;
            }
            std::string num = json_array.substr(start, pos - start);
            if (!num.empty()) {
                result.push_back(num);
            }
        }
    }

    return result;
}

void Server::send_json_response(int client_fd, int status_code, const std::string& json, bool keep_alive) {
    char header[512];
    int header_len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type,Authorization\r\n"
        "Connection: %s\r\n"
        "Keep-Alive: timeout=30, max=100\r\n"
        "\r\n",
        status_code, get_status_text(status_code).c_str(), json.size(),
        keep_alive ? "keep-alive" : "close");

    struct iovec iov[2];
    iov[0].iov_base = header;
    iov[0].iov_len = static_cast<size_t>(header_len);
    iov[1].iov_base = const_cast<char*>(json.data());
    iov[1].iov_len = json.size();
    writev(client_fd, iov, 2);
}

void Server::send_error(int client_fd, const std::string& error, bool keep_alive) {
    std::string json = "{\"error\":{\"message\":\"" + error + "\",\"type\":\"invalid_request_error\",\"code\":400}}";
    send_json_response(client_fd, 400, json, keep_alive);
}

std::string Server::get_status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default: return "OK";
    }
}
