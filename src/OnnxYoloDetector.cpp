#include "OnnxYoloDetector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef IVDS_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace {
constexpr std::array<const char*, 6> kNeuClasses = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"};

std::string className(int classId) {
    if (classId >= 0 && classId < static_cast<int>(kNeuClasses.size())) {
        return kNeuClasses[static_cast<size_t>(classId)];
    }
    return "class_" + std::to_string(classId);
}

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

LetterboxResult letterbox(const cv::Mat& frame, int inputSize) {
    const float scale = std::min(
        static_cast<float>(inputSize) / static_cast<float>(frame.cols),
        static_cast<float>(inputSize) / static_cast<float>(frame.rows));
    const int resizedW = static_cast<int>(std::round(static_cast<float>(frame.cols) * scale));
    const int resizedH = static_cast<int>(std::round(static_cast<float>(frame.rows) * scale));
    const int padX = (inputSize - resizedW) / 2;
    const int padY = (inputSize - resizedH) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resizedW, resizedH));

    cv::Mat canvas(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, resizedW, resizedH)));

    LetterboxResult result;
    result.image = canvas;
    result.scale = scale;
    result.padX = static_cast<float>(padX);
    result.padY = static_cast<float>(padY);
    return result;
}

std::vector<float> makeBlob(const cv::Mat& image) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> blob;
    blob.reserve(static_cast<size_t>(image.rows * image.cols * 3));
    for (const cv::Mat& channel : channels) {
        blob.insert(blob.end(), channel.ptr<float>(), channel.ptr<float>() + channel.total());
    }
    return blob;
}
}

struct OnnxYoloDetector::Impl {
#ifdef IVDS_ENABLE_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "IndustrialVisionDefectSystem"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> inputNamesOwned;
    std::vector<std::string> outputNamesOwned;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
#endif
    AppConfig config;
    bool isAvailable = false;
};

OnnxYoloDetector::OnnxYoloDetector(const std::string& modelPath, const AppConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;

#ifdef IVDS_ENABLE_ONNX
    if (modelPath.empty()) {
        throw std::runtime_error("ONNX backend requires --model <path>.");
    }

    impl_->sessionOptions.SetIntraOpNumThreads(1);
    impl_->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env,
        wideModelPath.c_str(),
        impl_->sessionOptions);

    const size_t inputCount = impl_->session->GetInputCount();
    const size_t outputCount = impl_->session->GetOutputCount();
    for (size_t i = 0; i < inputCount; ++i) {
        auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
        impl_->inputNamesOwned.emplace_back(name.get());
    }
    for (size_t i = 0; i < outputCount; ++i) {
        auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
        impl_->outputNamesOwned.emplace_back(name.get());
    }
    for (const std::string& name : impl_->inputNamesOwned) {
        impl_->inputNames.push_back(name.c_str());
    }
    for (const std::string& name : impl_->outputNamesOwned) {
        impl_->outputNames.push_back(name.c_str());
    }

    impl_->isAvailable = true;
#else
    (void)modelPath;
    throw std::runtime_error(
        "This executable was built without ONNX Runtime. Rebuild with IVDS_ENABLE_ONNX=ON.");
#endif
}

OnnxYoloDetector::~OnnxYoloDetector() = default;

DetectionResult OnnxYoloDetector::detect(const cv::Mat& frame) const {
#ifndef IVDS_ENABLE_ONNX
    (void)frame;
    throw std::runtime_error("ONNX Runtime backend is not enabled in this build.");
#else
    if (!impl_->session) {
        throw std::runtime_error("ONNX Runtime session is not initialized.");
    }

    const int inputSize = impl_->config.runtime.inputSize;
    const LetterboxResult lb = letterbox(frame, inputSize);
    std::vector<float> inputTensorValues = makeBlob(lb.image);
    std::array<int64_t, 4> inputShape = {1, 3, inputSize, inputSize};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size());

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        impl_->inputNames.data(),
        &inputTensor,
        1,
        impl_->outputNames.data(),
        impl_->outputNames.size());

    if (outputs.empty()) {
        throw std::runtime_error("ONNX model returned no outputs.");
    }

    float* data = outputs[0].GetTensorMutableData<float>();
    const std::vector<int64_t> shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3) {
        throw std::runtime_error("Unsupported YOLO output shape. Expected 3 dimensions.");
    }

    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];
    bool channelsFirst = dim1 < dim2;
    int64_t channels = channelsFirst ? dim1 : dim2;
    int64_t candidates = channelsFirst ? dim2 : dim1;
    if (channels < 5) {
        throw std::runtime_error("Unsupported YOLO output channels.");
    }

    const int classCount = static_cast<int>(channels - 4);
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int64_t i = 0; i < candidates; ++i) {
        auto at = [&](int64_t channel) -> float {
            return channelsFirst
                ? data[channel * candidates + i]
                : data[i * channels + channel];
        };

        int bestClass = -1;
        float bestScore = 0.0f;
        for (int cls = 0; cls < classCount; ++cls) {
            const float score = at(4 + cls);
            if (score > bestScore) {
                bestScore = score;
                bestClass = cls;
            }
        }

        if (bestScore < impl_->config.runtime.confidenceThreshold) {
            continue;
        }

        const float cx = at(0);
        const float cy = at(1);
        const float w = at(2);
        const float h = at(3);

        const float left = (cx - w * 0.5f - lb.padX) / lb.scale;
        const float top = (cy - h * 0.5f - lb.padY) / lb.scale;
        const float right = (cx + w * 0.5f - lb.padX) / lb.scale;
        const float bottom = (cy + h * 0.5f - lb.padY) / lb.scale;

        const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(left))));
        const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(top))));
        const int x2 = std::max(0, std::min(frame.cols, static_cast<int>(std::round(right))));
        const int y2 = std::max(0, std::min(frame.rows, static_cast<int>(std::round(bottom))));
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        static_cast<float>(impl_->config.runtime.confidenceThreshold),
        static_cast<float>(impl_->config.runtime.nmsThreshold),
        kept);

    DetectionResult result;
    result.binaryMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (int index : kept) {
        Defect defect;
        defect.bbox = boxes[static_cast<size_t>(index)];
        defect.center = cv::Point2f(
            static_cast<float>(defect.bbox.x + defect.bbox.width * 0.5),
            static_cast<float>(defect.bbox.y + defect.bbox.height * 0.5));
        defect.area = static_cast<double>(defect.bbox.area());
        defect.aspectRatio = defect.bbox.height > 0
            ? static_cast<double>(defect.bbox.width) / static_cast<double>(defect.bbox.height)
            : 0.0;
        defect.confidence = scores[static_cast<size_t>(index)];
        defect.label = className(classIds[static_cast<size_t>(index)]);
        cv::rectangle(result.binaryMask, defect.bbox, cv::Scalar(255), -1);
        result.defects.push_back(defect);
    }

    return result;
#endif
}

bool OnnxYoloDetector::available() const {
    return impl_ && impl_->isAvailable;
}

std::string OnnxYoloDetector::backendName() const {
    return available() ? "onnxruntime-yolo" : "onnxruntime-unavailable";
}
