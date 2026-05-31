#include "InspectionReport.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace {
std::string escapeCsv(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}
}

void InspectionReport::add(InspectionRecord record) {
    records_.push_back(std::move(record));
}

bool InspectionReport::empty() const {
    return records_.empty();
}

void InspectionReport::writeCsv(const std::string& path) const {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write report: " + path);
    }

    file << "source_path,result_path,status,defect_count,gt_count,tp,fp,fn,max_area,total_area,mean_matched_iou,processing_ms\n";
    file << std::fixed << std::setprecision(3);
    for (const InspectionRecord& record : records_) {
        file << escapeCsv(record.sourcePath) << ','
             << escapeCsv(record.resultPath) << ','
             << record.status << ','
             << record.defectCount << ','
             << record.gtCount << ','
             << record.truePositive << ','
             << record.falsePositive << ','
             << record.falseNegative << ','
             << record.maxArea << ','
             << record.totalArea << ','
             << record.meanMatchedIou << ','
             << record.processingMs << '\n';
    }
}

void InspectionReport::writeSummary(const std::string& path) const {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write summary: " + path);
    }

    int okCount = 0;
    int ngCount = 0;
    int defectCount = 0;
    int gtCount = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double totalProcessingMs = 0.0;
    double weightedIouSum = 0.0;

    for (const InspectionRecord& record : records_) {
        if (record.status == "OK") {
            ++okCount;
        } else {
            ++ngCount;
        }
        defectCount += record.defectCount;
        gtCount += record.gtCount;
        tp += record.truePositive;
        fp += record.falsePositive;
        fn += record.falseNegative;
        totalProcessingMs += record.processingMs;
        weightedIouSum += record.meanMatchedIou * static_cast<double>(record.truePositive);
    }

    const int total = static_cast<int>(records_.size());
    const double avgMs = total > 0 ? totalProcessingMs / static_cast<double>(total) : 0.0;
    const double precision = (tp + fp) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
    const double recall = (tp + fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
    const double f1 = (precision + recall) > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
    const double meanIou = tp > 0 ? weightedIouSum / static_cast<double>(tp) : 0.0;

    file << std::fixed << std::setprecision(4);
    file << "Industrial Vision Defect System - Inspection Summary\n";
    file << "====================================================\n";
    file << "total_images: " << total << '\n';
    file << "ok_images: " << okCount << '\n';
    file << "ng_images: " << ngCount << '\n';
    file << "detected_defects: " << defectCount << '\n';
    file << "ground_truth_boxes: " << gtCount << '\n';
    file << "true_positive: " << tp << '\n';
    file << "false_positive: " << fp << '\n';
    file << "false_negative: " << fn << '\n';
    file << "precision: " << precision << '\n';
    file << "recall: " << recall << '\n';
    file << "f1_score: " << f1 << '\n';
    file << "mean_matched_iou: " << meanIou << '\n';
    file << "avg_processing_ms: " << avgMs << '\n';
}
