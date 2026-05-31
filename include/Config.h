#pragma once

#include "Types.h"

#include <string>

class Config {
public:
    static AppConfig load(const std::string& path);
    static void applyCliOverrides(AppConfig& config, int argc, char** argv);
    static void printUsage(const char* executableName);

private:
    static bool readBoolArg(const std::string& value);
};
