#include "Config.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template <typename T>
void readIfPresent(const cv::FileNode& node, const std::string& key, T& value) {
    if (!node.empty() && !node[key].empty()) {
        node[key] >> value;
    }
}
}

AppConfig Config::load(const std::string& path) {
    AppConfig config;

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Warning: cannot open config file: " << path
                  << ". Using built-in defaults.\n";
        return config;
    }

    const cv::FileNode preprocess = fs["preprocess"];
    readIfPresent(preprocess, "blur_ksize", config.preprocess.blurKsize);
    readIfPresent(preprocess, "threshold_method", config.preprocess.thresholdMethod);
    readIfPresent(preprocess, "binary_threshold", config.preprocess.binaryThreshold);
    readIfPresent(preprocess, "invert_binary", config.preprocess.invertBinary);
    readIfPresent(preprocess, "morph_kernel_size", config.preprocess.morphKernelSize);
    readIfPresent(preprocess, "open_iterations", config.preprocess.openIterations);
    readIfPresent(preprocess, "close_iterations", config.preprocess.closeIterations);

    const cv::FileNode detector = fs["detector"];
    readIfPresent(detector, "min_area", config.detector.minArea);
    readIfPresent(detector, "max_area", config.detector.maxArea);
    readIfPresent(detector, "min_aspect_ratio", config.detector.minAspectRatio);
    readIfPresent(detector, "max_aspect_ratio", config.detector.maxAspectRatio);
    readIfPresent(detector, "min_circularity", config.detector.minCircularity);
    readIfPresent(detector, "roi_x", config.detector.roiX);
    readIfPresent(detector, "roi_y", config.detector.roiY);
    readIfPresent(detector, "roi_width", config.detector.roiWidth);
    readIfPresent(detector, "roi_height", config.detector.roiHeight);

    const cv::FileNode runtime = fs["runtime"];
    readIfPresent(runtime, "input_path", config.runtime.inputPath);
    readIfPresent(runtime, "output_dir", config.runtime.outputDir);
    readIfPresent(runtime, "label_dir", config.runtime.labelDir);
    readIfPresent(runtime, "backend", config.runtime.backend);
    readIfPresent(runtime, "model_path", config.runtime.modelPath);
    readIfPresent(runtime, "use_camera", config.runtime.useCamera);
    readIfPresent(runtime, "camera_id", config.runtime.cameraId);
    readIfPresent(runtime, "show_window", config.runtime.showWindow);
    readIfPresent(runtime, "save_results", config.runtime.saveResults);
    readIfPresent(runtime, "demo_mode", config.runtime.demoMode);
    readIfPresent(runtime, "frame_delay_ms", config.runtime.frameDelayMs);
    readIfPresent(runtime, "evaluate", config.runtime.evaluate);
    readIfPresent(runtime, "iou_threshold", config.runtime.iouThreshold);
    readIfPresent(runtime, "input_size", config.runtime.inputSize);
    readIfPresent(runtime, "confidence_threshold", config.runtime.confidenceThreshold);
    readIfPresent(runtime, "nms_threshold", config.runtime.nmsThreshold);

    return config;
}

void Config::applyCliOverrides(AppConfig& config, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto nextValue = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + option);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--input") {
            config.runtime.inputPath = nextValue(arg);
            config.runtime.demoMode = false;
            config.runtime.useCamera = false;
        } else if (arg == "--config") {
            nextValue(arg);
        } else if (arg == "--camera") {
            config.runtime.cameraId = std::stoi(nextValue(arg));
            config.runtime.useCamera = true;
            config.runtime.demoMode = false;
        } else if (arg == "--output") {
            config.runtime.outputDir = nextValue(arg);
        } else if (arg == "--label-dir") {
            config.runtime.labelDir = nextValue(arg);
            config.runtime.evaluate = true;
        } else if (arg == "--backend") {
            config.runtime.backend = nextValue(arg);
        } else if (arg == "--model") {
            config.runtime.modelPath = nextValue(arg);
        } else if (arg == "--input-size") {
            config.runtime.inputSize = std::stoi(nextValue(arg));
        } else if (arg == "--conf") {
            config.runtime.confidenceThreshold = std::stod(nextValue(arg));
        } else if (arg == "--nms") {
            config.runtime.nmsThreshold = std::stod(nextValue(arg));
        } else if (arg == "--evaluate") {
            config.runtime.evaluate = readBoolArg(nextValue(arg));
        } else if (arg == "--iou") {
            config.runtime.iouThreshold = std::stod(nextValue(arg));
        } else if (arg == "--show") {
            config.runtime.showWindow = readBoolArg(nextValue(arg));
        } else if (arg == "--save") {
            config.runtime.saveResults = readBoolArg(nextValue(arg));
        } else if (arg == "--delay-ms") {
            config.runtime.frameDelayMs = std::stoi(nextValue(arg));
        } else if (arg == "--demo") {
            config.runtime.demoMode = true;
            config.runtime.useCamera = false;
            config.runtime.inputPath.clear();
        } else if (arg == "--threshold") {
            config.preprocess.binaryThreshold = std::stod(nextValue(arg));
            config.preprocess.thresholdMethod = "binary";
        } else if (arg == "--min-area") {
            config.detector.minArea = std::stod(nextValue(arg));
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
}

void Config::printUsage(const char* executableName) {
    std::cout
        << "Usage:\n"
        << "  " << executableName << " [--input image_or_video] [--camera 0] [--demo]\n\n"
        << "Options:\n"
        << "  --config <path>      YAML config file path. Default: configs/config.yaml.\n"
        << "  --input <path>       Image or video file path.\n"
        << "  --camera <id>        Open a camera stream.\n"
        << "  --demo               Run a generated industrial defect sample.\n"
        << "  --output <dir>       Directory for saved results.\n"
        << "  --label-dir <dir>    YOLO label directory for evaluation.\n"
        << "  --backend <name>     traditional, onnx, or hybrid.\n"
        << "  --model <path>       ONNX model path for onnx/hybrid backend.\n"
        << "  --input-size <int>   YOLO input size. Default: 640.\n"
        << "  --conf <value>       YOLO confidence threshold.\n"
        << "  --nms <value>        YOLO NMS threshold.\n"
        << "  --evaluate <0|1>     Compare detections with YOLO labels when available.\n"
        << "  --iou <value>        IoU threshold for TP/FP/FN evaluation.\n"
        << "  --show <0|1>         Show OpenCV window.\n"
        << "  --save <0|1>         Save annotated result images.\n"
        << "  --delay-ms <int>     Delay for folder playback when --show 1.\n"
        << "  --threshold <value>  Use fixed binary threshold.\n"
        << "  --min-area <value>   Minimum contour area for defect candidates.\n";
}

bool Config::readBoolArg(const std::string& value) {
    return value == "1" || value == "true" || value == "True" || value == "TRUE" ||
           value == "yes" || value == "on";
}
