cmake_minimum_required(VERSION 3.7)
project(ImageProcessing)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCV REQUIRED)


set(SOURCE_FILES main.cpp include.hpp sample_code.hpp sample_code.cpp plot.hpp plot.cpp)
add_executable(ImageProcessing ${SOURCE_FILES})
target_link_libraries(ImageProcessing ${OpenCV_LIBS})
