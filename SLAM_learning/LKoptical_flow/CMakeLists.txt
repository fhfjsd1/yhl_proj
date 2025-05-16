cmake_minimum_required(VERSION 3.2)
project(LK)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LK LK.cpp)
target_link_libraries(LK ${OpenCV_LIBS})