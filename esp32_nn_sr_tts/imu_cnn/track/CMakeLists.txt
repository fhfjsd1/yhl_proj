cmake_minimum_required(VERSION 3.10)
project(RectangleCenterDetection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(track track.cpp)

target_link_libraries(track ${OpenCV_LIBS})