#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct PreprocessConfig {
    int blurKsize = 5;
    std::string thresholdMethod = "otsu";
    double binaryThreshold = 80.0;
    bool invertBinary = true;
    int morphKernelSize = 5;
    int openIterations = 1;
    int closeIterations = 2;
};

struct DetectorConfig {
    double minArea = 80.0;
    double maxArea = 50000.0;
    double minAspectRatio = 0.05;
    double maxAspectRatio = 20.0;
    double minCircularity = 0.0;
    int roiX = -1;
    int roiY = -1;
    int roiWidth = 0;
    int roiHeight = 0;
};

struct RuntimeConfig {
    std::string inputPath;
    std::string outputDir = "results";
    std::string labelDir;
    std::string backend = "traditional";
    std::string modelPath;
    bool useCamera = false;
    int cameraId = 0;
    bool showWindow = true;
    bool saveResults = true;
    bool demoMode = true;
    int frameDelayMs = 1;
    bool evaluate = true;
    double iouThreshold = 0.3;
    int inputSize = 640;
    double confidenceThreshold = 0.25;
    double nmsThreshold = 0.45;
};

struct AppConfig {
    PreprocessConfig preprocess;
    DetectorConfig detector;
    RuntimeConfig runtime;
};

struct Defect {
    cv::Rect bbox;
    cv::Point2f center;
    double area = 0.0;
    double aspectRatio = 0.0;
    double circularity = 0.0;
    double confidence = 1.0;
    std::string label = "defect";
};

struct DetectionResult {
    std::vector<Defect> defects;
    cv::Mat binaryMask;
    bool isNg() const { return !defects.empty(); }
};

struct GroundTruthBox {
    int classId = -1;
    std::string className;
    cv::Rect bbox;
};

struct EvaluationResult {
    int gtCount = 0;
    int truePositive = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    double meanMatchedIou = 0.0;
};
