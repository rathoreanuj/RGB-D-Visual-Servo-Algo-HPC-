#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "DataLoader.h"
#include "ImageProcessor.h"
#include "MHD.h"
#include "PoseEstimator.h"
#include "SimulatedAnnealing.h"
#include "Logger.h"
#include "PerformanceMetrics.h"

/**
 * Main program implementing RGB-D Image Processing Algorithm
 * for Target Recognition and Pose Estimation
 * 
 * Based on paper: DOI 10.3390/s20020430
 * 
 * This is a DETERMINISTIC, SINGLE-THREADED implementation
 * that serves as ground truth for HPC parallelization.
 */

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --rgb <path>        Path to RGB image (required)\n";
    std::cout << "  --depth <path>      Path to depth map (required)\n";
    std::cout << "  --model <path>      Path to 3D model (.obj or .ply) (required)\n";
    std::cout << "  --output <path>     Output directory (default: output/)\n";
    std::cout << "  --seed <int>        Random seed for determinism (default: 42)\n";
    std::cout << "  --iterations <int>  Max SA iterations (default: 500)\n";
    std::cout << "  --help              Show this help message\n";
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "RGB-D POSE ESTIMATION - Ground Truth" << std::endl;
    std::cout << "Paper: DOI 10.3390/s20020430" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Parse command-line arguments
    std::string rgbPath, depthPath, modelPath;
    std::string outputDir = "output/";
    unsigned int randomSeed = 42;
    int maxIterations = 500;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rgb" && i + 1 < argc) {
            rgbPath = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depthPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            randomSeed = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Validate required arguments
    if (rgbPath.empty() || depthPath.empty() || modelPath.empty()) {
        std::cerr << "Error: Missing required arguments\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        // Set up logging and performance tracking
        Logger::setOutputDirectory(outputDir);
        PerformanceMetrics::reset();
        
        // Start total execution timer
        PerformanceMetrics::startTimer("total");
        
        // ===== PART 1: DATA LOADING =====
        std::cout << "\n[1/6] Loading data..." << std::endl;
        PerformanceMetrics::startTimer("data_loading");
        
        cv::Mat rgbImage = DataLoader::loadRGBImage(rgbPath);
        cv::Mat depthMap = DataLoader::loadDepthMap(depthPath);
        
        WireframeModel model;
        if (modelPath.find(".obj") != std::string::npos) {
            model = DataLoader::loadOBJModel(modelPath);
        } else if (modelPath.find(".ply") != std::string::npos) {
            model = DataLoader::loadPLYModel(modelPath);
        } else {
            throw std::runtime_error("Unsupported model format. Use .obj or .ply");
        }
        
        PerformanceMetrics::stopTimer("data_loading");
        
        // ===== PART 2: IMAGE PREPROCESSING =====
        std::cout << "\n[2/6] Preprocessing image..." << std::endl;
        PerformanceMetrics::startTimer("preprocessing");
        
        // Convert to grayscale
        cv::Mat grayscale = ImageProcessor::convertToGrayscale(rgbImage);
        Logger::saveImage(grayscale, "original_grayscale.png");
        
        // Compute original histogram (Equation 6)
        std::vector<double> originalHist;
        ImageProcessor::computeHistogram(grayscale, originalHist);
        
        // Perform histogram equalization (Equations 7-8)
        cv::Mat equalized = ImageProcessor::histogramEqualization(grayscale);
        Logger::saveImage(equalized, "equalized_grayscale.png");
        
        // Compute equalized histogram
        std::vector<double> equalizedHist;
        ImageProcessor::computeHistogram(equalized, equalizedHist);
        
        // Log histogram data for Plot 1
        Logger::logHistogramData(originalHist, equalizedHist);
        
        PerformanceMetrics::stopTimer("preprocessing");
        
        // ===== PART 3: IMAGE SEGMENTATION =====
        std::cout << "\n[3/6] Segmenting target object..." << std::endl;
        PerformanceMetrics::startTimer("segmentation");
        
        // Binary segmentation
        cv::Mat binary = ImageProcessor::segmentByThreshold(equalized);
        Logger::saveImage(binary, "segmented_binary.png");
        
        // Extract target point set T (Section 3.3)
        std::vector<cv::Point2d> targetPoints;
        ImageProcessor::extractTargetPoints(binary, targetPoints, true);
        
        if (targetPoints.empty()) {
            throw std::runtime_error("No target points extracted. Check segmentation.");
        }
        
        // Visualize target points
        cv::Mat targetViz = rgbImage.clone();
        for (const auto& pt : targetPoints) {
            cv::circle(targetViz, pt, 1, cv::Scalar(0, 255, 0), -1);
        }
        Logger::saveImage(targetViz, "target_points.png");
        
        PerformanceMetrics::stopTimer("segmentation");
        
        // ===== PART 4: SET UP CAMERA AND POSE ESTIMATOR =====
        std::cout << "\n[4/6] Setting up camera model..." << std::endl;
        PerformanceMetrics::startTimer("camera_setup");
        
        PoseEstimator poseEstimator;
        
        // Camera intrinsics (adjust based on your camera/image)
        // For a typical 640x480 image with ~60° FOV
        double fx = 500.0;  // Focal length X (pixels)
        double fy = 500.0;  // Focal length Y (pixels)
        double cx = rgbImage.cols / 2.0;  // Principal point X
        double cy = rgbImage.rows / 2.0;  // Principal point Y
        
        poseEstimator.setCameraIntrinsics(fx, fy, cx, cy);
        std::cout << "Camera intrinsics: fx=" << fx << ", fy=" << fy 
                  << ", cx=" << cx << ", cy=" << cy << std::endl;
        
        PerformanceMetrics::stopTimer("camera_setup");
        
        // ===== PART 5: POSE OPTIMIZATION USING SIMULATED ANNEALING =====
        std::cout << "\n[5/6] Optimizing 6D pose using Simulated Annealing..." << std::endl;
        PerformanceMetrics::startTimer("optimization");
        
        // Configure SA optimizer
        SimulatedAnnealing::SAConfig saConfig;
        saConfig.randomSeed = randomSeed;
        saConfig.maxIterations = maxIterations;
        saConfig.initialTemperature = 100.0;
        saConfig.coolingRate = 0.98;  // Slower cooling = more iterations
        
        // Define search space based on expected pose range
        saConfig.alpha_min = -M_PI;
        saConfig.alpha_max = M_PI;
        saConfig.beta_min = -M_PI;
        saConfig.beta_max = M_PI;
        saConfig.gamma_min = -M_PI;
        saConfig.gamma_max = M_PI;
        saConfig.tx_min = -rgbImage.cols / 2.0;
        saConfig.tx_max = rgbImage.cols / 2.0;
        saConfig.ty_min = -rgbImage.rows / 2.0;
        saConfig.ty_max = rgbImage.rows / 2.0;
        saConfig.tz_min = 200.0;   // Min depth (mm)
        saConfig.tz_max = 1500.0;  // Max depth (mm)
        
        SimulatedAnnealing sa(saConfig);
        
        // Track MHD computation count
        long long mhdComputationCount = 0;
        
        // Define objective function: MHD between projected model and target points
        auto objectiveFunction = [&](const Pose6D& pose) -> double {
            auto iterStart = std::chrono::high_resolution_clock::now();
            mhdComputationCount++;
            // Project 3D model to 2D using current pose
            auto projectedEdges = poseEstimator.projectModel(model, pose);
            
            if (projectedEdges.empty()) {
                return 1e6;  // Large penalty if no edges projected
            }
            
            // Compute MHD (Equations 9-11)
            double mhd = MHD::computeMHD(targetPoints, projectedEdges);
            
            auto iterEnd = std::chrono::high_resolution_clock::now();
            auto iterDuration = std::chrono::duration_cast<std::chrono::microseconds>(iterEnd - iterStart);
            PerformanceMetrics::recordIterationTime(iterDuration.count() / 1000.0);
            
            return mhd;
        };
        
        // Initial pose guess (can be random or heuristic)
        Pose6D initialPose;
        initialPose.alpha = 0.0;
        initialPose.beta = 0.0;
        initialPose.gamma = 0.0;
        initialPose.tx = 0.0;
        initialPose.ty = 0.0;
        initialPose.tz = 500.0;  // Initial depth guess
        
        double initialScore = objectiveFunction(initialPose);
        std::cout << "Initial pose MHD score: " << initialScore << std::endl;
        
        // Run optimization
        std::vector<double> convergenceLog;
        Pose6D finalPose = sa.optimize(objectiveFunction, initialPose, convergenceLog);
        
        double finalScore = objectiveFunction(finalPose);
        
        PerformanceMetrics::stopTimer("optimization");
        
        // ===== PART 6: LOGGING AND VISUALIZATION =====
        std::cout << "\n[6/6] Saving results..." << std::endl;
        PerformanceMetrics::startTimer("logging");
        
        // Log convergence data for Plot 2
        Logger::logConvergenceData(convergenceLog);
        
        // Log final pose for Plot 3
        Logger::logFinalPose(finalPose, finalScore);
        
        // Visualize final pose overlay
        auto finalProjectedEdges = poseEstimator.projectModel(model, finalPose);
        cv::Mat finalViz = rgbImage.clone();
        
        for (const auto& edge : finalProjectedEdges) {
            cv::line(finalViz, edge.first, edge.second, cv::Scalar(0, 0, 255), 2);
        }
        Logger::saveImage(finalViz, "final_pose_overlay.png");
        
        // Print summary
        double convergenceRate = 0.0;
        if (initialScore > 0) {
            convergenceRate = (initialScore - finalScore) / initialScore;
        }
        Logger::logSummary(convergenceLog.size(), initialScore, finalScore, convergenceRate);
        
        PerformanceMetrics::stopTimer("logging");
        PerformanceMetrics::stopTimer("total");
        
        // Calculate memory usage
        size_t rgbBytes = rgbImage.total() * rgbImage.elemSize();
        size_t depthBytes = depthMap.total() * depthMap.elemSize();
        size_t modelBytes = (model.vertices.size() * sizeof(cv::Point3d)) + 
                           (model.edges.size() * sizeof(std::pair<int, int>));
        size_t targetPointsBytes = targetPoints.size() * sizeof(cv::Point2d);
        PerformanceMetrics::estimateMemoryUsage(rgbBytes, depthBytes, modelBytes, targetPointsBytes);
        
        // Set algorithm metrics
        PerformanceMetrics::setAlgorithmMetrics(
            convergenceLog.size(), initialScore, finalScore,
            targetPoints.size(), model.edges.size(), mhdComputationCount
        );
        
        // Calculate throughput
        PerformanceMetrics::calculateThroughput();
        
        // Save performance metrics
        PerformanceMetrics::saveMetricsToJSON(outputDir + "/data/performance_metrics.json");
        PerformanceMetrics::saveMetricsToCSV(outputDir + "/data/performance_metrics.csv");
        
        // Print performance report
        PerformanceMetrics::printMetrics();
        
        std::cout << "\n✓ All results saved to: " << outputDir << std::endl;
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "1. Run: python scripts/plot_histogram.py" << std::endl;
        std::cout << "2. Run: python scripts/plot_convergence.py" << std::endl;
        std::cout << "3. Run: python scripts/plot_final_pose.py" << std::endl;
        std::cout << "4. Run: python scripts/plot_performance.py  (Performance Analysis)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
