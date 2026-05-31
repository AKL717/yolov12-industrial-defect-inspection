#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

class OnnxYoloDetector {
public:
    OnnxYoloDetector(const std::string& modelPath, const AppConfig& config);
    ~OnnxYoloDetector();

    OnnxYoloDetector(const OnnxYoloDetector&) = delete;
    OnnxYoloDetector& operator=(const OnnxYoloDetector&) = delete;

    DetectionResult detect(const cv::Mat& frame) const;

    bool available() const;
    std::string backendName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
