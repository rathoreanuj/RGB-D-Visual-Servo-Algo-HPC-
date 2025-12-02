cmake_minimum_required(VERSION 3.10)
project(RGBD_PoseEstimation VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for deterministic builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
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

# Create executable
add_executable(rgbd_pose_estimation ${SOURCES})

# Link libraries
target_link_libraries(rgbd_pose_estimation
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Enable OpenMP with SIMD support and advanced optimization flags
if(MSVC)
    target_compile_options(rgbd_pose_estimation PRIVATE 
        /openmp                    # Enable OpenMP
        /O2                        # Optimization level 2
        /arch:AVX2                 # Enable AVX2 SIMD instructions
        /fp:fast                   # Fast floating point
        /Qvec-report:2             # Vectorization report
    )
else()
    target_compile_options(rgbd_pose_estimation PRIVATE 
        -fopenmp                   # Enable OpenMP
        -O3                        # Highest optimization
        -march=native              # Use CPU-specific SIMD
        -ffast-math                # Fast math operations
        -ftree-vectorize           # Auto-vectorization
        -fopt-info-vec             # Vectorization report
    )
endif()

# Output directory
set_target_properties(rgbd_pose_estimation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
