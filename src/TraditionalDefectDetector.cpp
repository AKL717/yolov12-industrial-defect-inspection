#include "TraditionalDefectDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

TraditionalDefectDetector::TraditionalDefectDetector(AppConfig config)
    : config_(std::move(config)) {}

DetectionResult TraditionalDefectDetector::detect(const cv::Mat& frame) const {
    DetectionResult result;
    result.binaryMask = preprocess(frame);

    if (config_.detector.roiWidth > 0 && config_.detector.roiHeight > 0) {
        const cv::Rect imageRect(0, 0, result.binaryMask.cols, result.binaryMask.rows);
        const cv::Rect roi(
            config_.detector.roiX,
            config_.detector.roiY,
            config_.detector.roiWidth,
            config_.detector.roiHeight);
        const cv::Rect clippedRoi = roi & imageRect;

        cv::Mat roiMask = cv::Mat::zeros(result.binaryMask.size(), result.binaryMask.type());
        if (clippedRoi.area() > 0) {
            result.binaryMask(clippedRoi).copyTo(roiMask(clippedRoi));
        }
        result.binaryMask = roiMask;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(result.binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area <= 0.0) {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        const double perimeter = cv::arcLength(contour, true);
        const double circularity = perimeter > 1e-6
            ? 4.0 * CV_PI * area / (perimeter * perimeter)
            : 0.0;

        Defect defect;
        defect.bbox = box;
        defect.center = cv::Point2f(
            static_cast<float>(box.x + box.width * 0.5),
            static_cast<float>(box.y + box.height * 0.5));
        defect.area = area;
        defect.aspectRatio = box.height > 0
            ? static_cast<double>(box.width) / static_cast<double>(box.height)
            : 0.0;
        defect.circularity = circularity;

        if (passFilters(defect)) {
            result.defects.push_back(defect);
        }
    }

    std::sort(result.defects.begin(), result.defects.end(), [](const Defect& a, const Defect& b) {
        return a.area > b.area;
    });

    return result;
}

cv::Mat TraditionalDefectDetector::preprocess(const cv::Mat& frame) const {
    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame.clone();
    } else {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, enhanced);

    int blurKsize = std::max(1, config_.preprocess.blurKsize);
    if (blurKsize % 2 == 0) {
        ++blurKsize;
    }
    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, cv::Size(blurKsize, blurKsize), 0.0);

    cv::Mat binary;
    const int binaryType = config_.preprocess.invertBinary
        ? cv::THRESH_BINARY_INV
        : cv::THRESH_BINARY;

    if (config_.preprocess.thresholdMethod == "adaptive") {
        cv::adaptiveThreshold(
            blurred,
            binary,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            binaryType,
            31,
            5);
    } else if (config_.preprocess.thresholdMethod == "binary") {
        cv::threshold(
            blurred,
            binary,
            config_.preprocess.binaryThreshold,
            255,
            binaryType);
    } else {
        cv::threshold(
            blurred,
            binary,
            0,
            255,
            binaryType | cv::THRESH_OTSU);
    }

    const int kernelSize = std::max(1, config_.preprocess.morphKernelSize);
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(kernelSize, kernelSize));

    if (config_.preprocess.openIterations > 0) {
        cv::morphologyEx(
            binary,
            binary,
            cv::MORPH_OPEN,
            kernel,
            cv::Point(-1, -1),
            config_.preprocess.openIterations);
    }

    if (config_.preprocess.closeIterations > 0) {
        cv::morphologyEx(
            binary,
            binary,
            cv::MORPH_CLOSE,
            kernel,
            cv::Point(-1, -1),
            config_.preprocess.closeIterations);
    }

    return binary;
}

bool TraditionalDefectDetector::passFilters(const Defect& defect) const {
    const DetectorConfig& cfg = config_.detector;
    return defect.area >= cfg.minArea &&
           defect.area <= cfg.maxArea &&
           defect.aspectRatio >= cfg.minAspectRatio &&
           defect.aspectRatio <= cfg.maxAspectRatio &&
           defect.circularity >= cfg.minCircularity;
}
