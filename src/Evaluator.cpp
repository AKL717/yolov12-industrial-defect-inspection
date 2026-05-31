#include "Evaluator.h"

#include <algorithm>
#include <numeric>

EvaluationResult Evaluator::evaluate(
    const std::vector<Defect>& detections,
    const std::vector<GroundTruthBox>& groundTruths,
    double iouThreshold) {
    EvaluationResult result;
    result.gtCount = static_cast<int>(groundTruths.size());

    std::vector<bool> gtMatched(groundTruths.size(), false);
    double matchedIouSum = 0.0;

    for (const Defect& detection : detections) {
        double bestIou = 0.0;
        int bestIndex = -1;

        for (size_t i = 0; i < groundTruths.size(); ++i) {
            if (gtMatched[i]) {
                continue;
            }

            const double currentIou = iou(detection.bbox, groundTruths[i].bbox);
            if (currentIou > bestIou) {
                bestIou = currentIou;
                bestIndex = static_cast<int>(i);
            }
        }

        if (bestIndex >= 0 && bestIou >= iouThreshold) {
            gtMatched[static_cast<size_t>(bestIndex)] = true;
            ++result.truePositive;
            matchedIouSum += bestIou;
        } else {
            ++result.falsePositive;
        }
    }

    result.falseNegative = static_cast<int>(
        std::count(gtMatched.begin(), gtMatched.end(), false));
    result.meanMatchedIou = result.truePositive > 0
        ? matchedIouSum / static_cast<double>(result.truePositive)
        : 0.0;

    return result;
}

double Evaluator::iou(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    const double intersectionArea = static_cast<double>(intersection.area());
    const double unionArea = static_cast<double>(a.area() + b.area()) - intersectionArea;
    return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
}
