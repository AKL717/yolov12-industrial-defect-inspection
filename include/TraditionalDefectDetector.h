#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

class TraditionalDefectDetector {
public:
    explicit TraditionalDefectDetector(AppConfig config);

    DetectionResult detect(const cv::Mat& frame) const;

private:
    AppConfig config_;

    cv::Mat preprocess(const cv::Mat& frame) const;
    bool passFilters(const Defect& defect) const;
};
