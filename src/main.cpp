#include "Config.h"
#include "Evaluator.h"
#include "InspectionReport.h"
#include "OnnxYoloDetector.h"
#include "TraditionalDefectDetector.h"
#include "Visualizer.h"
#include "YoloLabelReader.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string findConfigPath(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--config") {
            return argv[i + 1];
        }
    }
    return "configs/config.yaml";
}

cv::Mat createDemoFrame() {
    cv::Mat image(720, 1080, CV_8UC3, cv::Scalar(230, 232, 228));

    cv::rectangle(image, cv::Rect(130, 110, 820, 480), cv::Scalar(208, 216, 212), -1);
    cv::rectangle(image, cv::Rect(130, 110, 820, 480), cv::Scalar(130, 145, 140), 3);

    cv::line(image, cv::Point(300, 260), cv::Point(670, 305), cv::Scalar(35, 35, 35), 9, cv::LINE_AA);
    cv::ellipse(image, cv::Point(760, 405), cv::Size(70, 28), 18, 0, 360, cv::Scalar(42, 42, 42), -1);
    cv::circle(image, cv::Point(455, 430), 32, cv::Scalar(55, 55, 55), -1);

    cv::putText(
        image,
        "Demo Workpiece",
        cv::Point(130, 660),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(80, 80, 80),
        2,
        cv::LINE_AA);

    return image;
}

bool isImageFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

InspectionRecord makeRecord(
    const std::string& sourcePath,
    const std::string& resultPath,
    const DetectionResult& result,
    double processingMs,
    const EvaluationResult* evaluation) {
    InspectionRecord record;
    record.sourcePath = sourcePath;
    record.resultPath = resultPath;
    record.status = result.isNg() ? "NG" : "OK";
    record.defectCount = static_cast<int>(result.defects.size());
    record.processingMs = processingMs;

    if (evaluation != nullptr) {
        record.gtCount = evaluation->gtCount;
        record.truePositive = evaluation->truePositive;
        record.falsePositive = evaluation->falsePositive;
        record.falseNegative = evaluation->falseNegative;
        record.meanMatchedIou = evaluation->meanMatchedIou;
    }

    for (const Defect& defect : result.defects) {
        record.totalArea += defect.area;
        record.maxArea = std::max(record.maxArea, defect.area);
    }

    return record;
}

InspectionRecord processSingleFrame(
    const cv::Mat& frame,
    const std::function<DetectionResult(const cv::Mat&)>& detect,
    const AppConfig& config,
    int frameIndex,
    double fps,
    const std::string& sourcePath,
    const std::string& outputPrefix,
    int displayDelayMs = -1) {
    const auto begin = std::chrono::steady_clock::now();
    DetectionResult result = detect(frame);
    const auto end = std::chrono::steady_clock::now();
    const double processingMs = std::chrono::duration<double, std::milli>(end - begin).count();

    std::vector<GroundTruthBox> groundTruths;
    bool hasEvaluation = false;
    EvaluationResult evaluation;
    if (config.runtime.evaluate && sourcePath != "demo") {
        const std::filesystem::path labelPath = YoloLabelReader::inferLabelPath(
            sourcePath,
            config.runtime.labelDir);
        if (std::filesystem::exists(labelPath)) {
            groundTruths = YoloLabelReader::read(labelPath, frame.size());
            evaluation = Evaluator::evaluate(
                result.defects,
                groundTruths,
                config.runtime.iouThreshold);
            hasEvaluation = true;
        }
    }

    cv::Mat annotated = hasEvaluation
        ? Visualizer::drawComparison(frame, result, groundTruths, evaluation, fps)
        : Visualizer::drawDetections(frame, result, fps);
    std::string resultPath;

    std::cout << "Frame " << frameIndex
              << " | status=" << (result.isNg() ? "NG" : "OK")
              << " | defects=" << result.defects.size()
              << " | time_ms=" << processingMs;
    if (hasEvaluation) {
        std::cout << " | gt=" << evaluation.gtCount
                  << " | tp=" << evaluation.truePositive
                  << " | fp=" << evaluation.falsePositive
                  << " | fn=" << evaluation.falseNegative;
    }
    std::cout << '\n';

    if (config.runtime.saveResults) {
        const std::string statusDir = config.runtime.outputDir + "/" + (result.isNg() ? "NG" : "OK");
        std::filesystem::create_directories(statusDir);
        resultPath = Visualizer::buildOutputPath(statusDir, outputPrefix, frameIndex);
        const std::string maskPath = Visualizer::buildOutputPath(config.runtime.outputDir, "mask", frameIndex);
        cv::imwrite(resultPath, annotated);
        cv::imwrite(maskPath, result.binaryMask);
    }

    if (config.runtime.showWindow) {
        cv::imshow("Industrial Vision Defect System", annotated);
        cv::imshow("Binary Mask", result.binaryMask);
        const int key = cv::waitKey(displayDelayMs < 0 ? 0 : displayDelayMs);
        if (key == 27 || key == 'q' || key == 'Q') {
            throw std::runtime_error("Stopped by user.");
        }
    }

    return makeRecord(sourcePath, resultPath, result, processingMs, hasEvaluation ? &evaluation : nullptr);
}

void processStream(
    cv::VideoCapture& capture,
    const std::function<DetectionResult(const cv::Mat&)>& detect,
    const AppConfig& config) {
    int frameIndex = 0;
    auto lastTick = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - lastTick).count();
        lastTick = now;
        const double fps = seconds > 1e-6 ? 1.0 / seconds : 0.0;

        DetectionResult result = detect(frame);
        cv::Mat annotated = Visualizer::drawDetections(frame, result, fps);

        std::cout << "Frame " << frameIndex
                  << " | status=" << (result.isNg() ? "NG" : "OK")
                  << " | defects=" << result.defects.size()
                  << " | fps=" << fps
                  << '\n';

        if (config.runtime.saveResults && result.isNg()) {
            std::filesystem::create_directories(config.runtime.outputDir);
            const std::string resultPath = Visualizer::buildOutputPath(config.runtime.outputDir, "ng", frameIndex);
            cv::imwrite(resultPath, annotated);
        }

        if (config.runtime.showWindow) {
            cv::imshow("Industrial Vision Defect System", annotated);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        ++frameIndex;
    }
}

void processImageDirectory(
    const std::string& inputDir,
    const std::function<DetectionResult(const cv::Mat&)>& detect,
    const AppConfig& config) {
    std::vector<std::filesystem::path> imagePaths;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && isImageFile(entry.path().string())) {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        throw std::runtime_error("No supported images found in directory: " + inputDir);
    }

    InspectionReport report;
    int index = 0;
    for (const auto& imagePath : imagePaths) {
        const cv::Mat image = cv::imread(imagePath.string());
        if (image.empty()) {
            std::cerr << "Warning: skip unreadable image: " << imagePath.string() << '\n';
            continue;
        }

        const std::string prefix = imagePath.stem().string();
        report.add(processSingleFrame(
            image,
            detect,
            config,
            index,
            0.0,
            imagePath.string(),
            prefix,
            config.runtime.frameDelayMs));
        ++index;
    }

    const std::string reportPath = config.runtime.outputDir + "/inspection_report.csv";
    report.writeCsv(reportPath);
    report.writeSummary(config.runtime.outputDir + "/inspection_summary.txt");
    std::cout << "Report saved: " << reportPath << '\n';
}

DetectionResult mergeResults(const DetectionResult& a, const DetectionResult& b) {
    DetectionResult merged = a;
    merged.defects.insert(merged.defects.end(), b.defects.begin(), b.defects.end());

    if (!a.binaryMask.empty() && !b.binaryMask.empty()) {
        cv::bitwise_or(a.binaryMask, b.binaryMask, merged.binaryMask);
    } else if (!b.binaryMask.empty()) {
        merged.binaryMask = b.binaryMask.clone();
    }

    return merged;
}
}

int main(int argc, char** argv) {
    try {
        AppConfig config = Config::load(findConfigPath(argc, argv));
        Config::applyCliOverrides(config, argc, argv);

        if (config.runtime.demoMode && config.detector.roiWidth <= 0) {
            config.detector.roiX = 130;
            config.detector.roiY = 110;
            config.detector.roiWidth = 820;
            config.detector.roiHeight = 480;
        }

        auto traditionalDetector = std::make_shared<TraditionalDefectDetector>(config);
        std::shared_ptr<OnnxYoloDetector> onnxDetector;
        const std::string backend = config.runtime.backend;

        if (backend == "onnx" || backend == "hybrid") {
            onnxDetector = std::make_shared<OnnxYoloDetector>(config.runtime.modelPath, config);
        } else if (backend != "traditional") {
            throw std::runtime_error("Unknown backend: " + backend);
        }

        std::function<DetectionResult(const cv::Mat&)> detect;
        if (backend == "onnx") {
            detect = [onnxDetector](const cv::Mat& frame) {
                return onnxDetector->detect(frame);
            };
        } else if (backend == "hybrid") {
            detect = [traditionalDetector, onnxDetector](const cv::Mat& frame) {
                return mergeResults(traditionalDetector->detect(frame), onnxDetector->detect(frame));
            };
        } else {
            detect = [traditionalDetector](const cv::Mat& frame) {
                return traditionalDetector->detect(frame);
            };
        }

        if (config.runtime.demoMode || (!config.runtime.useCamera && config.runtime.inputPath.empty())) {
            InspectionReport report;
            report.add(processSingleFrame(createDemoFrame(), detect, config, 0, 0.0, "demo", "demo"));
            report.writeCsv(config.runtime.outputDir + "/inspection_report.csv");
            report.writeSummary(config.runtime.outputDir + "/inspection_summary.txt");
            return 0;
        }

        if (config.runtime.useCamera) {
            cv::VideoCapture capture(config.runtime.cameraId);
            if (!capture.isOpened()) {
                throw std::runtime_error("Cannot open camera id: " + std::to_string(config.runtime.cameraId));
            }
            processStream(capture, detect, config);
            return 0;
        }

        if (std::filesystem::is_directory(config.runtime.inputPath)) {
            processImageDirectory(config.runtime.inputPath, detect, config);
            return 0;
        }

        if (isImageFile(config.runtime.inputPath)) {
            const cv::Mat image = cv::imread(config.runtime.inputPath);
            if (image.empty()) {
                throw std::runtime_error("Cannot read image: " + config.runtime.inputPath);
            }
            InspectionReport report;
            report.add(processSingleFrame(
                image,
                detect,
                config,
                0,
                0.0,
                config.runtime.inputPath,
                std::filesystem::path(config.runtime.inputPath).stem().string()));
            report.writeCsv(config.runtime.outputDir + "/inspection_report.csv");
            report.writeSummary(config.runtime.outputDir + "/inspection_summary.txt");
            return 0;
        }

        cv::VideoCapture capture(config.runtime.inputPath);
        if (!capture.isOpened()) {
            throw std::runtime_error("Cannot open video: " + config.runtime.inputPath);
        }
        processStream(capture, detect, config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        Config::printUsage(argv[0]);
        return 1;
    }

    return 0;
}
