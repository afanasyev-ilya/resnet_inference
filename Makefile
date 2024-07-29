# Project name
TARGET = ./infer.bin

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I./cxxopts/include -I/usr/local/cuda/include -I/usr/local/TensorRT-10.2.0/include -I/usr/include/opencv4 `pkg-config --cflags opencv4`
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/local/TensorRT-10.2.0/lib -L/usr/lib/x86_64-linux-gnu `pkg-config --libs opencv4` -lnvinfer -lcudart

# Source files
SRCS = resnet50_inference.cpp
OBJS = $(SRCS:.cpp=.o)
OBJDIR = ./obj

TRTEXEC = ../bin/trtexec
ONNX_MODEL = data/ResNet50.onnx
ENGINE_FP32 = data/ResNet50_fp32.engine
ENGINE_FP16 = data/ResNet50_fp16.engine
ENGINE_INT8 = data/ResNet50_int8.engine

# cxxopts repository
CXXOPTS_REPO = https://github.com/jarro2783/cxxopts.git

# Default target
all: submodule $(TARGET)

# Rule for cloning the cxxopts submodule
submodule:
	@if [ ! -d "cxxopts" ]; then \
		echo "Cloning cxxopts submodule..."; \
		git submodule add $(CXXOPTS_REPO) cxxopts; \
		git submodule update --init --recursive; \
	else \
		echo "Updating cxxopts submodule..."; \
		cd cxxopts && git pull origin master; \
	fi

# Create the obj directory if it doesn't exist
$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Rule for building the target
$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Rule for compiling source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for generating TensorRT engines
$(ENGINE_FP32): $(ONNX_MODEL)
	$(TRTEXEC) --onnx=$(ONNX_MODEL) --saveEngine=$(ENGINE_FP32)

$(ENGINE_FP16): $(ONNX_MODEL)
	$(TRTEXEC) --onnx=$(ONNX_MODEL) --saveEngine=$(ENGINE_FP16) --fp16

$(ENGINE_INT8): $(ONNX_MODEL)
	$(TRTEXEC) --onnx=$(ONNX_MODEL) --saveEngine=$(ENGINE_INT8) --int8

# Build all targets, including engines
build_all: $(ENGINE_FP32) $(ENGINE_FP16) $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) $(OBJS)
	rm -rf cxxopts
	rm -rf $(OBJDIR)
	rm -f $(ENGINE_FP32) $(ENGINE_FP16) $(ENGINE_INT8)

.PHONY: all clean submodule build_all