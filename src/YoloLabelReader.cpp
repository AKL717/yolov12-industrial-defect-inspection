#include "YoloLabelReader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {
int clampInt(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}
}

std::vector<GroundTruthBox> YoloLabelReader::read(
    const std::filesystem::path& labelPath,
    const cv::Size& imageSize) {
    std::vector<GroundTruthBox> boxes;

    std::ifstream file(labelPath);
    if (!file.is_open()) {
        return boxes;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int classId = -1;
        double cx = 0.0;
        double cy = 0.0;
        double width = 0.0;
        double height = 0.0;
        if (!(iss >> classId >> cx >> cy >> width >> height)) {
            continue;
        }

        const int boxWidth = static_cast<int>(std::round(width * imageSize.width));
        const int boxHeight = static_cast<int>(std::round(height * imageSize.height));
        const int x = static_cast<int>(std::round((cx - width * 0.5) * imageSize.width));
        const int y = static_cast<int>(std::round((cy - height * 0.5) * imageSize.height));

        const int left = clampInt(x, 0, imageSize.width - 1);
        const int top = clampInt(y, 0, imageSize.height - 1);
        const int right = clampInt(x + boxWidth, 0, imageSize.width);
        const int bottom = clampInt(y + boxHeight, 0, imageSize.height);

        GroundTruthBox box;
        box.classId = classId;
        box.className = className(classId);
        box.bbox = cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top));
        if (box.bbox.area() > 0) {
            boxes.push_back(box);
        }
    }

    return boxes;
}

std::filesystem::path YoloLabelReader::inferLabelPath(
    const std::filesystem::path& imagePath,
    const std::string& explicitLabelDir) {
    const std::filesystem::path labelName = imagePath.stem().string() + ".txt";
    if (!explicitLabelDir.empty()) {
        return std::filesystem::path(explicitLabelDir) / labelName;
    }

    const std::filesystem::path imageDir = imagePath.parent_path();
    if (imageDir.filename() == "images") {
        return imageDir.parent_path() / "labels" / labelName;
    }

    return imageDir / labelName;
}

std::string YoloLabelReader::className(int classId) {
    static const std::vector<std::string> neuClasses = {
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"};

    if (classId >= 0 && classId < static_cast<int>(neuClasses.size())) {
        return neuClasses[static_cast<size_t>(classId)];
    }

    return "class_" + std::to_string(classId);
}
