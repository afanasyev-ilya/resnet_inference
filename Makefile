# Project name
TARGET = ./infer

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I./cxxopts/include -I/usr/local/cuda/include -I/usr/local/TensorRT-10.2.0/include -I/usr/include/opencv4 `pkg-config --cflags opencv4`
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/local/TensorRT-10.2.0/lib -L/usr/lib/x86_64-linux-gnu `pkg-config --libs opencv4` -lnvinfer -lcudart

# Source files
SRCS = resnet50_inference.cpp
OBJS = $(SRCS:.cpp=.o)
OBJDIR = ./obj

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

# Clean rule
clean:
	rm -f $(TARGET) $(OBJS)
	rm -rf cxxopts
	rm -rf $(OBJDIR)