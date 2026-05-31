#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

class YoloLabelReader {
public:
    static std::vector<GroundTruthBox> read(
        const std::filesystem::path& labelPath,
        const cv::Size& imageSize);

    static std::filesystem::path inferLabelPath(
        const std::filesystem::path& imagePath,
        const std::string& explicitLabelDir);

private:
    static std::string className(int classId);
};
