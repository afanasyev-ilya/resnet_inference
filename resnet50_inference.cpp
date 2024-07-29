#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <vector>
#include <cxxopts.hpp>

using namespace nvinfer1;

constexpr int NUM_LABELS = 1000;
constexpr int IMAGE_C = 3;
constexpr int IMAGE_H = 224;
constexpr int IMAGE_W = 224;
constexpr int NUM_RUNS = 5;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

std::vector<char> loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    assert(file.good());
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

std::shared_ptr<ICudaEngine> loadEngine(const std::string& engineFile, Logger& logger) {
    std::vector<char> engineData = loadEngineFile(engineFile);
    std::shared_ptr<IRuntime> runtime{createInferRuntime(logger), [](IRuntime* r) {}};
    assert(runtime != nullptr);

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    return std::shared_ptr<ICudaEngine>(engine, [](ICudaEngine* e) {});
}

std::vector<float> preprocessImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        exit(1);
    }

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(IMAGE_H, IMAGE_W));

    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255);

    std::vector<float> inputData(IMAGE_C * IMAGE_H * IMAGE_W);
    for (int c = 0; c < IMAGE_C; ++c) {
        for (int h = 0; h < IMAGE_H; ++h) {
            for (int w = 0; w < IMAGE_W; ++w) {
                inputData[c * IMAGE_H * IMAGE_W + h * IMAGE_W + w] = resizedImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return inputData;
}


// Function to load category labels from a file
std::vector<std::string> loadCategoryLabels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open category labels file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    file.close();
    return labels;
}

// Function to get the category label from the scores vector
std::string getCategoryLabel(const std::vector<float>& scores) {
    auto labels = loadCategoryLabels("data/class_labels.txt");

    if (scores.size() != labels.size()) {
        std::cerr << "Scores and labels vector sizes do not match." << std::endl;
        exit(1);
    }

    int maxIndex = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
    std::cout << "max index: " << maxIndex << std::endl;

    return labels[maxIndex];
}

void printTopLabels(const std::vector<float>& scores, const int topN = NUM_RUNS) {
    auto labels = loadCategoryLabels("data/class_labels.txt");

    if (scores.size() != labels.size()) {
        std::cerr << "Scores and labels vector sizes do not match." << std::endl;
        exit(1);
    }

    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    std::cout << "Top " << topN << " categories with scores:" << std::endl;
    for (int i = 0; i < topN && i < indices.size(); ++i) {
        int idx = indices[i];
        std::cout << "Category: " << labels[idx] << ", Score: " << scores[idx] << std::endl;
    }
}

int main(int argc, char **argv) {
    cxxopts::Options options("ResNet50Inference", "ResNet50 inference using TensorRT with FP16 or FP32 precision");
    options.add_options()
        ("8,int8", "Enable INT8 precision")
        ("16,fp16", "Enable FP16 precision")
        ("32,fp32", "Enable FP32 precision (default)")
        ("e,engine", "Path to TensorRT engine file", cxxopts::value<std::string>())
        ("i,image", "Path to input image", cxxopts::value<std::string>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Determine precision
    bool useFP16 = result.count("fp16") > 0;
    bool useINT8 = result.count("int8") > 0;
    
    // Check for conflicting precision options
    if (useFP16 && useINT8) {
        std::cerr << "Error: Cannot use both FP16 and INT8 precision at the same time." << std::endl;
        return 1;
    }

    // Default to FP32 if neither FP16 nor INT8 is specified
    std::string engineFile;
    if (useINT8) {
        engineFile = result.count("engine") ? result["engine"].as<std::string>() : "data/ResNet50_int8.engine";
    } else if (useFP16) {
        engineFile = result.count("engine") ? result["engine"].as<std::string>() : "data/ResNet50_fp16.engine";
    } else {
        engineFile = result.count("engine") ? result["engine"].as<std::string>() : "data/ResNet50_fp32.engine";
    }

    if(useFP16) {
        std::cout << "using FP16\n";
    } else if(useINT8) {
        std::cout << "using INT8\n";
    } else {
        std::cout << "using FP32\n";
    }

    // get image path
    std::string imagePath = result.count("image") ? result["image"].as<std::string>() : "data/binoculars.jpeg";

    Logger logger;

    // Load the engine
    auto engine = loadEngine(engineFile, logger);
    assert(engine != nullptr);

    std::shared_ptr<IExecutionContext> context{engine->createExecutionContext(), [](IExecutionContext* c) {}};
    assert(context != nullptr);

    // Allocate buffers
    // Get binding indices
    int inputIndex = 0;
    int outputIndex = 1;

    assert(inputIndex != -1 && outputIndex != -1);

    size_t inputSize = IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float); // Assuming input dimensions are fixed
    size_t outputSize = NUM_LABELS * sizeof(float); // Assuming output size is NUM_LABELS classes

    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    // Prepare input data (fill with dummy data for this example)
    std::vector<float> inputData(IMAGE_C * IMAGE_H * IMAGE_W, 1.0f); // Replace with actual image preprocessing
    inputData = preprocessImage(imagePath);

    cudaMemcpy(buffers[inputIndex], inputData.data(), inputSize, cudaMemcpyHostToDevice);

    // Run inference

    for(int iter = 0; iter < 5; iter++)
    {
        // Start measuring time
        auto start = std::chrono::high_resolution_clock::now();
        // Execute the inference
        context->executeV2(buffers);
        // Stop measuring time
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate the duration
        std::chrono::duration<double> duration = end - start;
        // Print the execution time in milliseconds
        std::cout << "Inference time: " << duration.count() * NUM_LABELS << " ms" << std::endl;
    }

    // Copy output data from GPU to CPU
    std::vector<float> outputData(NUM_LABELS);
    cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    // Process the output (for example, find the class with the highest score)
    printTopLabels(outputData, 10);
    std::cout << "categoty: " << getCategoryLabel(outputData) << std::endl;
    

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}