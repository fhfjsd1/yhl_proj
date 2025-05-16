cmake_minimum_required(VERSION 3.5)
project(imgbasic)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_executable(img imgBasic.cpp)

target_link_libraries(img ${OpenCV_LIBS})