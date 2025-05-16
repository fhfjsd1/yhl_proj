cmake_minimum_required(VERSION 3.5)
project(joinmap)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu")

cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0167 NEW)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3/")

find_package(PCL REQUIRED COMPONENTS common io)
find_package(VTK REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(joinmap joinmap.cpp)
target_link_libraries(joinmap ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${VTK_LIBRARIES})