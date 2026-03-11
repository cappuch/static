.PHONY: all clean tiktoken embedder

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall
INCLUDES = -I. -Itiktoken-c

BUILD_DIR = build
DIST_DIR = build/dist

TIKTOKEN_LIB = $(BUILD_DIR)/libtiktoken_c.a
EMBEDDER_OBJS = $(BUILD_DIR)/embedder.o $(BUILD_DIR)/tokenizer_wrapper.o $(BUILD_DIR)/binary.o $(BUILD_DIR)/server.o $(BUILD_DIR)/main.o

all: $(BUILD_DIR) $(DIST_DIR) $(TIKTOKEN_LIB) $(DIST_DIR)/embedder

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

$(TIKTOKEN_LIB):
	cd tiktoken-c && cargo build --release --lib
	cp tiktoken-c/target/release/libtiktoken_c.a $@

$(DIST_DIR)/embedder: $(EMBEDDER_OBJS) $(TIKTOKEN_LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ -lpthread -ldl

$(BUILD_DIR)/embedder.o: src/embedder.cpp src/embedder.h src/binary.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ src/embedder.cpp

$(BUILD_DIR)/tokenizer_wrapper.o: src/tokenizer_wrapper.cpp src/tokenizer_wrapper.h tiktoken-c/tiktoken.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ src/tokenizer_wrapper.cpp

$(BUILD_DIR)/binary.o: src/binary.cpp src/binary.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ src/binary.cpp

$(BUILD_DIR)/server.o: src/server.cpp src/server.h src/embedder.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ src/server.cpp

$(BUILD_DIR)/main.o: src/main.cpp src/embedder.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ src/main.cpp

clean:
	rm -rf $(BUILD_DIR)
