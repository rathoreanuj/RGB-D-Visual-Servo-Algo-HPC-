cmake_minimum_required(VERSION 3.10)
project(RGBD_PoseEstimation VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

set(SOURCES
    src/main.cpp
    src/DataLoader.cpp
    src/ImageProcessor.cpp
    src/MHD.cpp
    src/SimulatedAnnealing.cpp
    src/PoseEstimator.cpp
    src/Logger.cpp
    src/PerformanceMetrics.cpp
)

add_executable(rgbd_pose_estimation ${SOURCES})

target_link_libraries(rgbd_pose_estimation
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(rgbd_pose_estimation PRIVATE 
        /openmp
        /O2
        /arch:AVX2
        /fp:fast
        /Qvec-report:2
    )
else()
    target_compile_options(rgbd_pose_estimation PRIVATE 
        -fopenmp
        -O3
        -march=native
        -ffast-math
        -ftree-vectorize
        -fopt-info-vec
    )
endif()

set_target_properties(rgbd_pose_estimation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
