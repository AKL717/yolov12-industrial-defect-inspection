#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <string>

class Visualizer {
public:
    static cv::Mat drawDetections(const cv::Mat& frame, const DetectionResult& result, double fps);
    static cv::Mat drawComparison(
        const cv::Mat& frame,
        const DetectionResult& result,
        const std::vector<GroundTruthBox>& groundTruths,
        const EvaluationResult& evaluation,
        double fps);
    static std::string buildOutputPath(const std::string& outputDir, const std::string& prefix, int index);
};
