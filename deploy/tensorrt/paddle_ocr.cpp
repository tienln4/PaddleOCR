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
static const int INPUT_W = 100;
static const int OUTPUT_SIZE = 25*38;
const char* INPUT_BLOB_NAME = "x";
const char* OUTPUT_BLOB_NAME = "save_infer_model/scale_0.tmp_1";
const std::string alphabet = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const std::string weight_file = "";
const std::string engine_file = "/paddle/model/plate_ocr.engine";
const std::string img_path = "test_plate.jpg";
using namespace nvinfer1;
static Logger gLogger;

std::vector<float> prepareImage(cv::Mat src_img) 
{
    int m_net_W = 100;
    int m_net_H = 32;
    std::vector<float> result(m_net_W * m_net_H * 3);
    float *data = result.data();
    int index = 0;

    cv::Mat flt_img = cv::Mat::zeros(cv::Size(m_net_W, m_net_H), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(src_img, rsz_img, cv::Size(m_net_W, m_net_H));
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255.);

    //HWC TO CHW
    int channelLength = m_net_W * m_net_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(m_net_H, m_net_W, CV_32FC1, data + channelLength * (index + 2)),
            cv::Mat(m_net_H, m_net_W, CV_32FC1, data + channelLength * (index + 1)),
            cv::Mat(m_net_H, m_net_W, CV_32FC1, data + channelLength * index)
    };
    index += 3;
    cv::split(flt_img, split_img);
    return result;
}

void CrnnResizeImg(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio) {
    int imgC, imgH, imgW;
    imgC = 3;
    imgH = 32;
    imgW = 100;

    imgW = int(32 * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
                cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                        int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                        {127, 127, 127});
}

void Normalize (cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &scale, const bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(*im, bgr_channels);
  for (auto i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i],
                              (0.0 - mean[i]) * scale[i]);
  }
  cv::merge(bgr_channels, *im);
}

void Permute (const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

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
    // for (auto i = 0; i < 10; i++) {
    //     std::cout << output[i] << " ";
    // }
    // std::cout << std::endl;
    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {

    std::string data_dir = "/paddle/X/";
    std::string gt_path = data_dir + "gt.txt";
    std::ifstream read_file;
    read_file.open (gt_path);
    std::string text;

    int num_file = 0;
    while (getline (read_file, text)) 
    {
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
      
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
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

        num_file ++;
        int colum = 0;
        std::string colum_val = "";
        std::string img_path = "";
        std::string gt = "";

        for (int i = 0; i < text.length(); i++)
        {
            if (text[i] == '\t' || i == text.length()-1)
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
        std::cout<< img_path << std::endl;

        // cv::Mat img = cv::imread(img_path);
        cv::Mat img = cv::imread("/paddle/images/08.jpg");
        cv::Mat resize_img;
        if (img.empty()) continue;
        // cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
        // for (int i = 0; i < INPUT_H * INPUT_W; i++) 
        // {
        //     data[i] = (img.at<cv::Vec3b>(i)[0]/255.0-0.5)/0.5;
        //     data[i + INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[1]/255.0-0.5)/0.5;
        //     data[i + 2 * INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[2]/255.0-0.5)/0.5;
        // }

        // std::vector<float> input =  prepareImage(img);
        // float wh_ratio = float(img.cols) / float(img.rows);
        // CrnnResizeImg(img, resize_img, wh_ratio);
        // Normalize(&resize_img, {0.5f, 0.5f, 0.5f}, {1 / 0.5f, 1 / 0.5f, 1 / 0.5f}, true);
        // std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
        // Permute(&resize_img, input.data());


        

        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[i] = (uc_pixel[2] / 255.0-0.5)/0.5;
                data[i + INPUT_H * INPUT_W] = (uc_pixel[1] / 255.0-0.5)/0.5;
                data[i + 2 * INPUT_H * INPUT_W] = (uc_pixel[0] / 255.0-0.5)/0.5;
                uc_pixel += 3;
                ++i;
                // exit(0);
            }
        }


        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<int> preds;
        for (int i = 0; i < 25; i++) {
            int maxj = 0;
            for (int j = 1; j < 38; j++) {
                if (prob[38 * i + j] > prob[38 * i + maxj]) maxj = j;
            }
            preds.push_back(maxj);
            // std::cout << maxj << " ";
        }
        // std::cout << std::endl;
        // std::cout << "raw_pred: " << strDecode(preds, true) << std::endl;
        std::cout << "pred: " << strDecode(preds, false)<< " " << "gt: " << gt<<std::endl;
        // if (num_file == 20) break;
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
        context->destroy();
        engine->destroy();
        runtime->destroy();
        exit(0);
    }
    
    read_file.close();
    
    return 0;
}
