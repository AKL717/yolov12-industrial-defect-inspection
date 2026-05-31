cmake_minimum_required(VERSION 3.16)

project(IndustrialVisionDefectSystem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

option(IVDS_ENABLE_ONNX "Build ONNX Runtime YOLO detector backend" OFF)

if (IVDS_ENABLE_ONNX)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root directory")
    if (NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers not found. Run scripts/download_onnxruntime.ps1 or set ONNXRUNTIME_ROOT.")
    endif()
    if (MSVC)
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
        set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    else()
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
endif()

add_executable(industrial_vision_defect
    src/Config.cpp
    src/Evaluator.cpp
    src/InspectionReport.cpp
    src/OnnxYoloDetector.cpp
    src/TraditionalDefectDetector.cpp
    src/Visualizer.cpp
    src/YoloLabelReader.cpp
    src/main.cpp
)

target_include_directories(industrial_vision_defect
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(industrial_vision_defect PRIVATE ${OpenCV_LIBS})

if (IVDS_ENABLE_ONNX)
    target_compile_definitions(industrial_vision_defect PRIVATE IVDS_ENABLE_ONNX)
    target_include_directories(industrial_vision_defect PRIVATE "${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(industrial_vision_defect PRIVATE "${ONNXRUNTIME_LIB}")
endif()

if (WIN32)
    get_filename_component(OpenCV_RUNTIME_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)
    file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_RUNTIME_DIR}/opencv_world*.dll")
    if (OpenCV_RUNTIME_DLLS)
        add_custom_command(TARGET industrial_vision_defect POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OpenCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:industrial_vision_defect>
            COMMENT "Copying OpenCV runtime DLLs")
    endif()
endif()

if (IVDS_ENABLE_ONNX AND WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
    add_custom_command(TARGET industrial_vision_defect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:industrial_vision_defect>
        COMMENT "Copying ONNX Runtime DLL")
endif()

if (MSVC)
    target_compile_options(industrial_vision_defect PRIVATE /W4 /EHsc)
else()
    target_compile_options(industrial_vision_defect PRIVATE -Wall -Wextra -Wpedantic)
endif()
