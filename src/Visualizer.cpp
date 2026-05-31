#include "Visualizer.h"

#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <sstream>

namespace {
void drawTextBackground(cv::Mat& canvas, const cv::Rect& rect, const cv::Scalar& color) {
    const cv::Rect clipped = rect & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (clipped.area() <= 0) {
        return;
    }
    cv::Mat roi = canvas(clipped);
    cv::Mat overlay(roi.size(), roi.type(), color);
    cv::addWeighted(overlay, 0.68, roi, 0.32, 0.0, roi);
}

void drawSmallLabel(
    cv::Mat& canvas,
    const std::string& text,
    const cv::Point& anchor,
    const cv::Scalar& textColor) {
    constexpr double fontScale = 0.38;
    constexpr int thickness = 1;
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

    int x = std::max(2, std::min(anchor.x, canvas.cols - textSize.width - 8));
    int y = std::max(textSize.height + 4, std::min(anchor.y, canvas.rows - 4));
    const cv::Rect bg(x, y - textSize.height - 4, textSize.width + 8, textSize.height + baseline + 6);
    drawTextBackground(canvas, bg, cv::Scalar(15, 15, 15));
    cv::putText(
        canvas,
        text,
        cv::Point(x + 4, y),
        cv::FONT_HERSHEY_SIMPLEX,
        fontScale,
        textColor,
        thickness,
        cv::LINE_AA);
}

void drawHud(cv::Mat& canvas, const std::string& text, const cv::Scalar& color) {
    constexpr double fontScale = 0.45;
    constexpr int thickness = 1;
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    const int height = textSize.height + baseline + 12;
    const int y = std::max(0, canvas.rows - height);
    drawTextBackground(canvas, cv::Rect(0, y, canvas.cols, height), cv::Scalar(15, 15, 15));
    cv::putText(
        canvas,
        text,
        cv::Point(8, y + textSize.height + 5),
        cv::FONT_HERSHEY_SIMPLEX,
        fontScale,
        color,
        thickness,
        cv::LINE_AA);
}
}

cv::Mat Visualizer::drawDetections(const cv::Mat& frame, const DetectionResult& result, double fps) {
    cv::Mat canvas;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, canvas, cv::COLOR_GRAY2BGR);
    } else {
        canvas = frame.clone();
    }
    cv::copyMakeBorder(canvas, canvas, 0, 28, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(18, 18, 18));

    const cv::Scalar okColor(60, 180, 75);
    const cv::Scalar ngColor(40, 40, 230);
    const cv::Scalar color = result.isNg() ? ngColor : okColor;
    const bool showBoxLabels = frame.cols >= 480 || frame.rows >= 480;

    for (size_t i = 0; i < result.defects.size(); ++i) {
        const Defect& defect = result.defects[i];
        cv::rectangle(canvas, defect.bbox, ngColor, 2);
        cv::circle(canvas, defect.center, 2, cv::Scalar(255, 180, 30), -1);

        std::ostringstream label;
        label << defect.label;
        if (defect.confidence < 0.999) {
            label << " " << std::fixed << std::setprecision(2) << defect.confidence;
        }
        if (showBoxLabels) {
            drawSmallLabel(canvas, label.str(), cv::Point(defect.bbox.x, defect.bbox.y - 3), cv::Scalar(255, 255, 255));
        }
    }

    std::ostringstream status;
    status << (result.isNg() ? "NG" : "OK")
           << " d=" << result.defects.size()
           << " fps=" << std::fixed << std::setprecision(1) << fps;
    drawHud(canvas, status.str(), color);

    return canvas;
}

cv::Mat Visualizer::drawComparison(
    const cv::Mat& frame,
    const DetectionResult& result,
    const std::vector<GroundTruthBox>& groundTruths,
    const EvaluationResult& evaluation,
    double fps) {
    cv::Mat canvas = drawDetections(frame, result, fps);
    const cv::Scalar gtColor(60, 190, 60);
    const bool showBoxLabels = frame.cols >= 480 || frame.rows >= 480;

    for (const GroundTruthBox& gt : groundTruths) {
        cv::rectangle(canvas, gt.bbox, gtColor, 2);

        const std::string label = "GT " + gt.className;
        if (showBoxLabels) {
            drawSmallLabel(canvas, label, cv::Point(gt.bbox.x, gt.bbox.y - 3), gtColor);
        }
    }

    std::ostringstream metrics;
    if (frame.cols < 260) {
        metrics << (result.isNg() ? "NG" : "OK")
                << " d" << result.defects.size()
                << " G" << evaluation.gtCount
                << " T" << evaluation.truePositive
                << " F" << evaluation.falsePositive;
    } else {
        metrics << (result.isNg() ? "NG" : "OK")
                << " d=" << result.defects.size()
                << " G=" << evaluation.gtCount
                << " T=" << evaluation.truePositive
                << " F=" << evaluation.falsePositive
                << " N=" << evaluation.falseNegative
                << " IoU=" << std::fixed << std::setprecision(2) << evaluation.meanMatchedIou;
    }
    drawHud(canvas, metrics.str(), cv::Scalar(230, 230, 230));

    return canvas;
}

std::string Visualizer::buildOutputPath(const std::string& outputDir, const std::string& prefix, int index) {
    std::ostringstream path;
    path << outputDir << "/" << prefix << "_" << std::setw(5) << std::setfill('0') << index << ".jpg";
    return path.str();
}
