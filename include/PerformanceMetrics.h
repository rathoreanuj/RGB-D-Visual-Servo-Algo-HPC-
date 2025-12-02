#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <string>
#include <chrono>
#include <map>
#include <vector>

/**
 * @brief Performance metrics tracker for HPC baseline measurement
 * 
 * Tracks key metrics:
 * 1. Execution time per module (data loading, preprocessing, segmentation, optimization)
 * 2. Algorithm convergence metrics (iterations, convergence rate, final score)
 * 3. Memory usage estimates
 * 4. Throughput metrics (points processed, MHD computations)
 * 5. CPU utilization (single-threaded baseline)
 */
class PerformanceMetrics {
public:
    struct Timer {
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        double elapsedMs;
        
        Timer() : elapsedMs(0.0) {}
    };
    
    struct ModuleTiming {
        double dataLoadingMs;
        double preprocessingMs;
        double segmentationMs;
        double cameraSetupMs;
        double optimizationMs;
        double loggingMs;
        double totalMs;
        
        ModuleTiming() : dataLoadingMs(0), preprocessingMs(0), segmentationMs(0),
                        cameraSetupMs(0), optimizationMs(0), loggingMs(0), totalMs(0) {}
    };
    
    struct AlgorithmMetrics {
        int totalIterations;
        double initialMHDScore;
        double finalMHDScore;
        double improvement;
        double convergenceRate;
        double avgIterationTimeMs;
        double bestIterationTimeMs;
        double worstIterationTimeMs;
        int numTargetPoints;
        int numModelEdges;
        long long totalMHDComputations;
        
        AlgorithmMetrics() : totalIterations(0), initialMHDScore(0), finalMHDScore(0),
                           improvement(0), convergenceRate(0), avgIterationTimeMs(0),
                           bestIterationTimeMs(1e9), worstIterationTimeMs(0),
                           numTargetPoints(0), numModelEdges(0), totalMHDComputations(0) {}
    };
    
    struct MemoryMetrics {
        size_t rgbImageBytes;
        size_t depthMapBytes;
        size_t modelBytes;
        size_t targetPointsBytes;
        size_t totalEstimatedBytes;
        
        MemoryMetrics() : rgbImageBytes(0), depthMapBytes(0), modelBytes(0),
                         targetPointsBytes(0), totalEstimatedBytes(0) {}
    };
    
    struct ThroughputMetrics {
        double pointsPerSecond;        // Target points processed/sec
        double edgesPerSecond;          // Model edges processed/sec
        double mhdComputationsPerSecond; // MHD calculations/sec
        double iterationsPerSecond;     // SA iterations/sec
        
        ThroughputMetrics() : pointsPerSecond(0), edgesPerSecond(0),
                             mhdComputationsPerSecond(0), iterationsPerSecond(0) {}
    };
    
    /**
     * @brief Start a timer for a specific module
     * @param moduleName Name of the module (e.g., "data_loading")
     */
    static void startTimer(const std::string& moduleName);
    
    /**
     * @brief Stop a timer and record elapsed time
     * @param moduleName Name of the module
     */
    static void stopTimer(const std::string& moduleName);
    
    /**
     * @brief Record iteration timing during optimization
     * @param iterationTimeMs Time taken for iteration in milliseconds
     */
    static void recordIterationTime(double iterationTimeMs);
    
    /**
     * @brief Set algorithm metrics
     */
    static void setAlgorithmMetrics(int iterations, double initialScore, double finalScore,
                                   int targetPoints, int modelEdges, long long mhdComputations);
    
    /**
     * @brief Estimate memory usage
     */
    static void estimateMemoryUsage(size_t rgbBytes, size_t depthBytes, 
                                   size_t modelBytes, size_t targetPointsBytes);
    
    /**
     * @brief Calculate throughput metrics
     */
    static void calculateThroughput();
    
    /**
     * @brief Print all metrics to console
     */
    static void printMetrics();
    
    /**
     * @brief Save metrics to JSON file
     * @param filepath Path to output JSON file
     */
    static void saveMetricsToJSON(const std::string& filepath);
    
    /**
     * @brief Save metrics to CSV file (for plotting)
     * @param filepath Path to output CSV file
     */
    static void saveMetricsToCSV(const std::string& filepath);
    
    /**
     * @brief Get module timings
     */
    static ModuleTiming getModuleTiming();
    
    /**
     * @brief Get algorithm metrics
     */
    static AlgorithmMetrics getAlgorithmMetrics();
    
    /**
     * @brief Get memory metrics
     */
    static MemoryMetrics getMemoryMetrics();
    
    /**
     * @brief Get throughput metrics
     */
    static ThroughputMetrics getThroughputMetrics();
    
    /**
     * @brief Reset all metrics
     */
    static void reset();

private:
    static std::map<std::string, Timer> timers_;
    static ModuleTiming moduleTiming_;
    static AlgorithmMetrics algorithmMetrics_;
    static MemoryMetrics memoryMetrics_;
    static ThroughputMetrics throughputMetrics_;
    static std::vector<double> iterationTimes_;
    
    static double getElapsedMs(const Timer& timer);
};

#endif // PERFORMANCE_METRICS_H
