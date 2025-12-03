#include "PerformanceMetrics.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
std::map<std::string, PerformanceMetrics::Timer> PerformanceMetrics::timers_;
PerformanceMetrics::ModuleTiming PerformanceMetrics::moduleTiming_;
PerformanceMetrics::AlgorithmMetrics PerformanceMetrics::algorithmMetrics_;
PerformanceMetrics::MemoryMetrics PerformanceMetrics::memoryMetrics_;
PerformanceMetrics::ThroughputMetrics PerformanceMetrics::throughputMetrics_;
std::vector<double> PerformanceMetrics::iterationTimes_;
void PerformanceMetrics::startTimer(const std::string& moduleName) {
    Timer timer;
    timer.start = std::chrono::high_resolution_clock::now();
    timers_[moduleName] = timer;
}
void PerformanceMetrics::stopTimer(const std::string& moduleName) {
    auto it = timers_.find(moduleName);
    if (it != timers_.end()) {
        it->second.end = std::chrono::high_resolution_clock::now();
        it->second.elapsedMs = getElapsedMs(it->second);
        if (moduleName == "data_loading") {
            moduleTiming_.dataLoadingMs = it->second.elapsedMs;
        } else if (moduleName == "preprocessing") {
            moduleTiming_.preprocessingMs = it->second.elapsedMs;
        } else if (moduleName == "segmentation") {
            moduleTiming_.segmentationMs = it->second.elapsedMs;
        } else if (moduleName == "camera_setup") {
            moduleTiming_.cameraSetupMs = it->second.elapsedMs;
        } else if (moduleName == "optimization") {
            moduleTiming_.optimizationMs = it->second.elapsedMs;
        } else if (moduleName == "logging") {
            moduleTiming_.loggingMs = it->second.elapsedMs;
        } else if (moduleName == "total") {
            moduleTiming_.totalMs = it->second.elapsedMs;
        }
    }
}
void PerformanceMetrics::recordIterationTime(double iterationTimeMs) {
    iterationTimes_.push_back(iterationTimeMs);
}
void PerformanceMetrics::setAlgorithmMetrics(int iterations, double initialScore, 
                                            double finalScore, int targetPoints, 
                                            int modelEdges, long long mhdComputations) {
    algorithmMetrics_.totalIterations = iterations;
    algorithmMetrics_.initialMHDScore = initialScore;
    algorithmMetrics_.finalMHDScore = finalScore;
    algorithmMetrics_.improvement = initialScore - finalScore;
    algorithmMetrics_.convergenceRate = (initialScore > 0) ? 
                                       (algorithmMetrics_.improvement / initialScore) : 0.0;
    algorithmMetrics_.numTargetPoints = targetPoints;
    algorithmMetrics_.numModelEdges = modelEdges;
    algorithmMetrics_.totalMHDComputations = mhdComputations;
    if (!iterationTimes_.empty()) {
        algorithmMetrics_.avgIterationTimeMs = 
            std::accumulate(iterationTimes_.begin(), iterationTimes_.end(), 0.0) / iterationTimes_.size();
        algorithmMetrics_.bestIterationTimeMs = 
            *std::min_element(iterationTimes_.begin(), iterationTimes_.end());
        algorithmMetrics_.worstIterationTimeMs = 
            *std::max_element(iterationTimes_.begin(), iterationTimes_.end());
    }
}
void PerformanceMetrics::estimateMemoryUsage(size_t rgbBytes, size_t depthBytes,
                                            size_t modelBytes, size_t targetPointsBytes) {
    memoryMetrics_.rgbImageBytes = rgbBytes;
    memoryMetrics_.depthMapBytes = depthBytes;
    memoryMetrics_.modelBytes = modelBytes;
    memoryMetrics_.targetPointsBytes = targetPointsBytes;
    memoryMetrics_.totalEstimatedBytes = rgbBytes + depthBytes + modelBytes + targetPointsBytes;
}
void PerformanceMetrics::calculateThroughput() {
    if (moduleTiming_.optimizationMs > 0) {
        double optimizationTimeSec = moduleTiming_.optimizationMs / 1000.0;
        throughputMetrics_.iterationsPerSecond = 
            algorithmMetrics_.totalIterations / optimizationTimeSec;
        throughputMetrics_.pointsPerSecond = 
            (algorithmMetrics_.numTargetPoints * algorithmMetrics_.totalIterations) / optimizationTimeSec;
        throughputMetrics_.edgesPerSecond = 
            (algorithmMetrics_.numModelEdges * algorithmMetrics_.totalIterations) / optimizationTimeSec;
        throughputMetrics_.mhdComputationsPerSecond = 
            algorithmMetrics_.totalMHDComputations / optimizationTimeSec;
    }
}
void PerformanceMetrics::printMetrics() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           PERFORMANCE METRICS REPORT                       ║" << std::endl;
    std::cout << "║           (Single-threaded Baseline for HPC)               ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 1. MODULE EXECUTION TIME (milliseconds)                 │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Data Loading:          " << std::setw(10) << std::fixed << std::setprecision(2) 
              << moduleTiming_.dataLoadingMs << " ms  (" 
              << std::setw(5) << std::setprecision(1) 
              << (moduleTiming_.dataLoadingMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "│ Preprocessing:         " << std::setw(10) << moduleTiming_.preprocessingMs << " ms  (" 
              << std::setw(5) << (moduleTiming_.preprocessingMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "│ Segmentation:          " << std::setw(10) << moduleTiming_.segmentationMs << " ms  (" 
              << std::setw(5) << (moduleTiming_.segmentationMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "│ Camera Setup:          " << std::setw(10) << moduleTiming_.cameraSetupMs << " ms  (" 
              << std::setw(5) << (moduleTiming_.cameraSetupMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "│ Optimization (SA):     " << std::setw(10) << moduleTiming_.optimizationMs << " ms  (" 
              << std::setw(5) << (moduleTiming_.optimizationMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "│ Logging/Visualization: " << std::setw(10) << moduleTiming_.loggingMs << " ms  (" 
              << std::setw(5) << (moduleTiming_.loggingMs / moduleTiming_.totalMs * 100) << "%) │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ TOTAL EXECUTION TIME:  " << std::setw(10) << moduleTiming_.totalMs << " ms           │" << std::endl;
    std::cout << "│                        " << std::setw(10) << (moduleTiming_.totalMs / 1000.0) 
              << " sec          │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 2. ALGORITHM PERFORMANCE METRICS                        │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Total Iterations:      " << std::setw(10) << algorithmMetrics_.totalIterations 
              << "                │" << std::endl;
    std::cout << "│ Initial MHD Score:     " << std::setw(10) << std::setprecision(4) 
              << algorithmMetrics_.initialMHDScore << "                │" << std::endl;
    std::cout << "│ Final MHD Score:       " << std::setw(10) << algorithmMetrics_.finalMHDScore 
              << "                │" << std::endl;
    std::cout << "│ Improvement:           " << std::setw(10) << algorithmMetrics_.improvement 
              << "                │" << std::endl;
    std::cout << "│ Convergence Rate:      " << std::setw(10) << std::setprecision(2) 
              << (algorithmMetrics_.convergenceRate * 100) << " %            │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Avg Iteration Time:    " << std::setw(10) << algorithmMetrics_.avgIterationTimeMs 
              << " ms           │" << std::endl;
    std::cout << "│ Best Iteration Time:   " << std::setw(10) << algorithmMetrics_.bestIterationTimeMs 
              << " ms           │" << std::endl;
    std::cout << "│ Worst Iteration Time:  " << std::setw(10) << algorithmMetrics_.worstIterationTimeMs 
              << " ms           │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 3. DATA SIZE METRICS                                    │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Target Points:         " << std::setw(10) << algorithmMetrics_.numTargetPoints 
              << "                │" << std::endl;
    std::cout << "│ Model Edges:           " << std::setw(10) << algorithmMetrics_.numModelEdges 
              << "                │" << std::endl;
    std::cout << "│ Total MHD Calls:       " << std::setw(10) << algorithmMetrics_.totalMHDComputations 
              << "                │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 4. MEMORY USAGE ESTIMATE                                │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ RGB Image:             " << std::setw(10) << (memoryMetrics_.rgbImageBytes / 1024.0) 
              << " KB           │" << std::endl;
    std::cout << "│ Depth Map:             " << std::setw(10) << (memoryMetrics_.depthMapBytes / 1024.0) 
              << " KB           │" << std::endl;
    std::cout << "│ 3D Model:              " << std::setw(10) << (memoryMetrics_.modelBytes / 1024.0) 
              << " KB           │" << std::endl;
    std::cout << "│ Target Points:         " << std::setw(10) << (memoryMetrics_.targetPointsBytes / 1024.0) 
              << " KB           │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Total Estimated:       " << std::setw(10) << (memoryMetrics_.totalEstimatedBytes / (1024.0 * 1024.0)) 
              << " MB           │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ 5. THROUGHPUT METRICS (items/second)                    │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Iterations/sec:        " << std::setw(10) << throughputMetrics_.iterationsPerSecond 
              << "                │" << std::endl;
    std::cout << "│ Points/sec:            " << std::setw(10) << std::setprecision(0) 
              << throughputMetrics_.pointsPerSecond << "                │" << std::endl;
    std::cout << "│ Edges/sec:             " << std::setw(10) << throughputMetrics_.edgesPerSecond 
              << "                │" << std::endl;
    std::cout << "│ MHD Calls/sec:         " << std::setw(10) << throughputMetrics_.mhdComputationsPerSecond 
              << "                │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  KEY BOTTLENECK: Optimization module (" 
              << std::setprecision(1) << (moduleTiming_.optimizationMs / moduleTiming_.totalMs * 100) 
              << "% of total time) ║" << std::endl;
    std::cout << "║  → Primary target for HPC parallelization               ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;
}
void PerformanceMetrics::saveMetricsToJSON(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing" << std::endl;
        return;
    }
    file << "{\n";
    file << "  \"module_timing\": {\n";
    file << "    \"data_loading_ms\": " << moduleTiming_.dataLoadingMs << ",\n";
    file << "    \"preprocessing_ms\": " << moduleTiming_.preprocessingMs << ",\n";
    file << "    \"segmentation_ms\": " << moduleTiming_.segmentationMs << ",\n";
    file << "    \"camera_setup_ms\": " << moduleTiming_.cameraSetupMs << ",\n";
    file << "    \"optimization_ms\": " << moduleTiming_.optimizationMs << ",\n";
    file << "    \"logging_ms\": " << moduleTiming_.loggingMs << ",\n";
    file << "    \"total_ms\": " << moduleTiming_.totalMs << ",\n";
    file << "    \"total_sec\": " << (moduleTiming_.totalMs / 1000.0) << "\n";
    file << "  },\n";
    file << "  \"algorithm_metrics\": {\n";
    file << "    \"total_iterations\": " << algorithmMetrics_.totalIterations << ",\n";
    file << "    \"initial_mhd_score\": " << algorithmMetrics_.initialMHDScore << ",\n";
    file << "    \"final_mhd_score\": " << algorithmMetrics_.finalMHDScore << ",\n";
    file << "    \"improvement\": " << algorithmMetrics_.improvement << ",\n";
    file << "    \"convergence_rate\": " << algorithmMetrics_.convergenceRate << ",\n";
    file << "    \"avg_iteration_time_ms\": " << algorithmMetrics_.avgIterationTimeMs << ",\n";
    file << "    \"best_iteration_time_ms\": " << algorithmMetrics_.bestIterationTimeMs << ",\n";
    file << "    \"worst_iteration_time_ms\": " << algorithmMetrics_.worstIterationTimeMs << ",\n";
    file << "    \"num_target_points\": " << algorithmMetrics_.numTargetPoints << ",\n";
    file << "    \"num_model_edges\": " << algorithmMetrics_.numModelEdges << ",\n";
    file << "    \"total_mhd_computations\": " << algorithmMetrics_.totalMHDComputations << "\n";
    file << "  },\n";
    file << "  \"memory_metrics\": {\n";
    file << "    \"rgb_image_bytes\": " << memoryMetrics_.rgbImageBytes << ",\n";
    file << "    \"depth_map_bytes\": " << memoryMetrics_.depthMapBytes << ",\n";
    file << "    \"model_bytes\": " << memoryMetrics_.modelBytes << ",\n";
    file << "    \"target_points_bytes\": " << memoryMetrics_.targetPointsBytes << ",\n";
    file << "    \"total_estimated_bytes\": " << memoryMetrics_.totalEstimatedBytes << ",\n";
    file << "    \"total_estimated_mb\": " << (memoryMetrics_.totalEstimatedBytes / (1024.0 * 1024.0)) << "\n";
    file << "  },\n";
    file << "  \"throughput_metrics\": {\n";
    file << "    \"iterations_per_second\": " << throughputMetrics_.iterationsPerSecond << ",\n";
    file << "    \"points_per_second\": " << throughputMetrics_.pointsPerSecond << ",\n";
    file << "    \"edges_per_second\": " << throughputMetrics_.edgesPerSecond << ",\n";
    file << "    \"mhd_computations_per_second\": " << throughputMetrics_.mhdComputationsPerSecond << "\n";
    file << "  }\n";
    file << "}\n";
    file.close();
    std::cout << "Performance metrics saved to: " << filepath << std::endl;
}
void PerformanceMetrics::saveMetricsToCSV(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing" << std::endl;
        return;
    }
    file << "metric_category,metric_name,value,unit\n";
    file << "timing,data_loading," << moduleTiming_.dataLoadingMs << ",ms\n";
    file << "timing,preprocessing," << moduleTiming_.preprocessingMs << ",ms\n";
    file << "timing,segmentation," << moduleTiming_.segmentationMs << ",ms\n";
    file << "timing,camera_setup," << moduleTiming_.cameraSetupMs << ",ms\n";
    file << "timing,optimization," << moduleTiming_.optimizationMs << ",ms\n";
    file << "timing,logging," << moduleTiming_.loggingMs << ",ms\n";
    file << "timing,total," << moduleTiming_.totalMs << ",ms\n";
    file << "algorithm,total_iterations," << algorithmMetrics_.totalIterations << ",count\n";
    file << "algorithm,initial_mhd_score," << algorithmMetrics_.initialMHDScore << ",score\n";
    file << "algorithm,final_mhd_score," << algorithmMetrics_.finalMHDScore << ",score\n";
    file << "algorithm,improvement," << algorithmMetrics_.improvement << ",score\n";
    file << "algorithm,convergence_rate," << algorithmMetrics_.convergenceRate << ",ratio\n";
    file << "algorithm,avg_iteration_time," << algorithmMetrics_.avgIterationTimeMs << ",ms\n";
    file << "throughput,iterations_per_second," << throughputMetrics_.iterationsPerSecond << ",iter/s\n";
    file << "throughput,points_per_second," << throughputMetrics_.pointsPerSecond << ",points/s\n";
    file << "throughput,edges_per_second," << throughputMetrics_.edgesPerSecond << ",edges/s\n";
    file << "throughput,mhd_computations_per_second," << throughputMetrics_.mhdComputationsPerSecond << ",calls/s\n";
    file.close();
    std::cout << "Performance metrics CSV saved to: " << filepath << std::endl;
}
PerformanceMetrics::ModuleTiming PerformanceMetrics::getModuleTiming() {
    return moduleTiming_;
}
PerformanceMetrics::AlgorithmMetrics PerformanceMetrics::getAlgorithmMetrics() {
    return algorithmMetrics_;
}
PerformanceMetrics::MemoryMetrics PerformanceMetrics::getMemoryMetrics() {
    return memoryMetrics_;
}
PerformanceMetrics::ThroughputMetrics PerformanceMetrics::getThroughputMetrics() {
    return throughputMetrics_;
}
void PerformanceMetrics::reset() {
    timers_.clear();
    moduleTiming_ = ModuleTiming();
    algorithmMetrics_ = AlgorithmMetrics();
    memoryMetrics_ = MemoryMetrics();
    throughputMetrics_ = ThroughputMetrics();
    iterationTimes_.clear();
}
double PerformanceMetrics::getElapsedMs(const Timer& timer) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(timer.end - timer.start);
    return duration.count() / 1000.0;
}
