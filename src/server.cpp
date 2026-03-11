#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "server.h"
#include "embedder.h"
#include "tokenizer_wrapper.h"

Server::Server(Embedder* embedder, uint16_t port)
    : embedder_(embedder)
    , tokenizer_(new Tokenizer(Tokenizer::Model::CL100K_BASE))
    , port_(port)
    , running_(false)
    , server_fd_(-1) {}

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

    if (listen(server_fd_, 10) < 0) {
        std::cerr << "Failed to listen" << std::endl;
        close(server_fd_);
        return;
    }

    running_ = true;
    std::cout << "Server started on http://0.0.0.0:" << port_ << std::endl;
    std::cout << "OpenAI-compatible endpoint: http://0.0.0.0:" << port_ << "/v1/embeddings" << std::endl;

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

        handle_request(client_fd);
        close(client_fd);
    }
}

void Server::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
}

void Server::handle_request(int client_fd) {
    char buffer[8192] = {0};
    ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
    
    if (bytes_read <= 0) {
        return;
    }

    std::string request(buffer, bytes_read);
    
    std::istringstream request_stream(request);
    std::string method, path, version;
    request_stream >> method >> path >> version;

    std::string headers;
    std::string line;
    std::getline(request_stream, line);
    while (std::getline(request_stream, line) && line != "\r") {
        headers += line + "\n";
    }

    std::string body;
    std::string content_length_str = get_header_value(headers, "Content-Length:");
    if (!content_length_str.empty()) {
        size_t content_length = std::stoul(content_length_str);
        size_t header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos && header_end + 4 + content_length <= request.size()) {
            body = request.substr(header_end + 4, content_length);
        }
    }

    if (path == "/v1/embeddings" && method == "POST") {
        handle_embeddings(client_fd, body);
    } else if (path == "/health" && method == "GET") {
        send_json_response(client_fd, 200, "{\"status\":\"ok\"}");
    } else {
        send_error(client_fd, "Not Found");
    }
}

std::string Server::get_header_value(const std::string& headers, const std::string& key) {
    std::istringstream stream(headers);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.length() >= key.length() && 
            line.substr(0, key.length()) == key) {
            std::string value = line.substr(key.length());
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_of(" \r"));
            return value;
        }
    }
    return "";
}

std::string Server::extract_json_body(const std::string& request) {
    size_t header_end = request.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        return request.substr(header_end + 4);
    }
    return "";
}

void Server::handle_embeddings(int client_fd, const std::string& body) {
    if (body.empty()) {
        send_error(client_fd, "Invalid request body");
        return;
    }

    std::string input = extract_json_value(body, "input");
    std::string model = extract_json_value(body, "model");
    
    if (input.empty()) {
        send_error(client_fd, "Missing 'input' field");
        return;
    }

    std::vector<std::string> texts;
    if (input[0] == '[') {
        texts = parse_json_array(input);
    } else {
        texts = {input};
    }

    if (texts.empty()) {
        send_error(client_fd, "Invalid 'input' field");
        return;
    }

    auto start_tokenize = std::chrono::high_resolution_clock::now();
    auto tokens = tokenizer_->encode(texts[0]);
    auto end_tokenize = std::chrono::high_resolution_clock::now();
    
    auto start_inference = std::chrono::high_resolution_clock::now();
    auto embeddings = embedder_->get_token_embeddings(texts);
    auto end_inference = std::chrono::high_resolution_clock::now();

    double tokenizing_time = std::chrono::duration<double, std::milli>(end_tokenize - start_tokenize).count();
    double inference_time = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();
    
    std::ostringstream response;
    response << "{";
    response << "\"object\":\"list\",";
    response << "\"data\":[";
    
    for (size_t i = 0; i < embeddings.size(); ++i) {
        response << "{";
        response << "\"object\":\"embedding\",";
        response << "\"embedding\":[";
        for (size_t j = 0; j < embeddings[i].size(); ++j) {
            response << embeddings[i][j];
            if (j < embeddings[i].size() - 1) response << ",";
        }
        response << "],";
        response << "\"index\":" << i << ",";
        response << "\"model\":\"" << (model.empty() ? "text-embedding-ada-002" : model) << "\"";
        response << "}";
        if (i < embeddings.size() - 1) response << ",";
    }
    
    response << "],";
    response << "\"model\":\"" << (model.empty() ? "text-embedding-ada-002" : model) << "\",";
    response << "\"usage\":{";
    response << "\"prompt_tokens\":" << 0 << ",";
    response << "\"total_tokens\":" << 0;
    response << "},";
    response << "\"tokenizing_time\":" << tokenizing_time << ",";
    response << "\"inference_time\":" << inference_time;
    response << "}";

    send_json_response(client_fd, 200, response.str());
}

std::string Server::extract_json_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t key_pos = json.find(search);
    if (key_pos == std::string::npos) return "";
    
    size_t colon_pos = json.find(":", key_pos);
    if (colon_pos == std::string::npos) return "";
    
    size_t value_start = colon_pos + 1;
    while (value_start < json.size() && (json[value_start] == ' ' || json[value_start] == '\t')) {
        value_start++;
    }
    
    if (value_start >= json.size()) return "";
    
    if (json[value_start] == '"') {
        value_start++;
        size_t value_end = json.find('"', value_start);
        if (value_end == std::string::npos) return "";
        return json.substr(value_start, value_end - value_start);
    }
    
    size_t value_end = value_start;
    while (value_end < json.size() && json[value_end] != ',' && json[value_end] != '}' && json[value_end] != ']' && json[value_end] != '\n') {
        value_end++;
    }
    return json.substr(value_start, value_end - value_start);
}

std::vector<std::string> Server::parse_json_array(const std::string& json_array) {
    std::vector<std::string> result;
    size_t pos = 0;
    
    while (pos < json_array.size()) {
        if (json_array[pos] == '"') {
            pos++;
            size_t end = json_array.find('"', pos);
            if (end == std::string::npos) break;
            result.push_back(json_array.substr(pos, end - pos));
            pos = end + 1;
        } else {
            pos++;
        }
    }
    
    return result;
}

void Server::send_json_response(int client_fd, int status_code, const std::string& json) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " " << get_status_text(status_code) << "\r\n";
    response << "Content-Type: application/json\r\n";
    response << "Content-Length: " << json.size() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    response << "\r\n";
    response << json;

    std::string response_str = response.str();
    write(client_fd, response_str.c_str(), response_str.size());
}

void Server::send_error(int client_fd, const std::string& error) {
    std::string json = "{\"error\":{\"message\":\"" + error + "\",\"type\":\"invalid_request_error\",\"code\":400}}";
    send_json_response(client_fd, 400, json);
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
