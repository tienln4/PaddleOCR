#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <fstream>
#include <string>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define USE_FP16 
#define DEVICE 0
#define BATCH_SIZE 1
static const int INPUT_H = 32;
static const int INPUT_W = 256;
static const int OUTPUT_SIZE = 65 * 37;
const char* INPUT_BLOB_NAME = "text";
const char* OUTPUT_BLOB_NAME = "Prediction";
const std::string alphabet = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const std::string engine_file = "/paddle/model/deep_text.engine";
const std::string img_path = "test_plate.jpg";
using namespace nvinfer1;
static Logger gLogger;


std::string strDecode(std::vector<int>& preds, bool raw) {
    std::string str;
    if (raw) {
        for (auto v: preds) {
            str.push_back(alphabet[v]);
        }
    } else {
        for (size_t i = 0; i < preds.size(); i++) {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i])) continue;
            str.push_back(alphabet[preds[i]]);
        }
    }
    return str;
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 1 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {

    //----------------------
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data ---------------------------
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 1 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::string data_dir = "/paddle/valid/";
    std::string gt_path = data_dir + "gt.txt";
    std::ifstream read_file;
    read_file.open (gt_path);
    std::string text;
    int num_file = 0;
    while (getline (read_file, text)) 
    {
        static float data[BATCH_SIZE * 1 * INPUT_H * INPUT_W];
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        num_file ++;
        int colum = 0;
        std::string colum_val = "";
        std::string img_path = "";
        std::string gt = "";

        for (int i = 0; i <= text.length(); i++)
        {
            if (text[i] == '\t' || i == text.length())
            {
                colum ++;
                if (colum == 1) img_path = colum_val;
                else gt = colum_val;
                colum_val = "";
            }
            else
            {
                colum_val = colum_val + text[i];
            }
        }
        img_path = data_dir + img_path;

        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = ((float)img.at<uchar>(i) / 255.0 - 0.5) * 2.0;
        }

        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<int> preds;
        for (int i = 0; i < 65; i++) {
            int maxj = 0;
            for (int j = 1; j < 37; j++) {
                if (prob[37 * i + j] > prob[37 * i + maxj]) maxj = j;
            }
            preds.push_back(maxj);
        }

        std::string result = "false";
        float time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (strDecode(preds, false) == gt)
        {
            result = "true";
        }
        std::cout << "pred: " << strDecode(preds, false)<< "      " << "gt: " << gt<< "       " << result << "     " << time <<std::endl;
        
    }
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    read_file.close();
    
    return 0;
}
