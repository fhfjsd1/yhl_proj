cmake_minimum_required(VERSION 3.2)
project(feture)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")


list(APPEND CMAKE_MODULE_PATH /home/taylor/Downloads/g2o-master/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

# add_executable(extraction extraction.cpp)
# target_link_libraries(extraction ${OpenCV_LIBS})

add_executable(2d_2d 2d_2d.cpp)
target_link_libraries(2d_2d 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    ${G2O_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_solver_csparse g2o_types_slam3d
)

