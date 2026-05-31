#pragma once

#include <string>
#include <vector>

struct InspectionRecord {
    std::string sourcePath;
    std::string resultPath;
    std::string status;
    int defectCount = 0;
    int gtCount = 0;
    int truePositive = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    double maxArea = 0.0;
    double totalArea = 0.0;
    double meanMatchedIou = 0.0;
    double processingMs = 0.0;
};

class InspectionReport {
public:
    void add(InspectionRecord record);
    bool empty() const;
    void writeCsv(const std::string& path) const;
    void writeSummary(const std::string& path) const;

private:
    std::vector<InspectionRecord> records_;
};
