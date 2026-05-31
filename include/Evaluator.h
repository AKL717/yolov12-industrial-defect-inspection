#pragma once

#include "Types.h"

#include <vector>

class Evaluator {
public:
    static EvaluationResult evaluate(
        const std::vector<Defect>& detections,
        const std::vector<GroundTruthBox>& groundTruths,
        double iouThreshold);

    static double iou(const cv::Rect& a, const cv::Rect& b);
};
