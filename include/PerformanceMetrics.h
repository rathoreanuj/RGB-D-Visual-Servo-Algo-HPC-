#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H
#include <string>
#include <chrono>
#include <map>
#include <vector>
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
        double pointsPerSecond;        
        double edgesPerSecond;          
        double mhdComputationsPerSecond; 
        double iterationsPerSecond;     
        ThroughputMetrics() : pointsPerSecond(0), edgesPerSecond(0),
                             mhdComputationsPerSecond(0), iterationsPerSecond(0) {}
    };
    static void startTimer(const std::string& moduleName);
    static void stopTimer(const std::string& moduleName);
    static void recordIterationTime(double iterationTimeMs);
    static void setAlgorithmMetrics(int iterations, double initialScore, double finalScore,
                                   int targetPoints, int modelEdges, long long mhdComputations);
    static void estimateMemoryUsage(size_t rgbBytes, size_t depthBytes, 
                                   size_t modelBytes, size_t targetPointsBytes);
    static void calculateThroughput();
    static void printMetrics();
    static void saveMetricsToJSON(const std::string& filepath);
    static void saveMetricsToCSV(const std::string& filepath);
    static ModuleTiming getModuleTiming();
    static AlgorithmMetrics getAlgorithmMetrics();
    static MemoryMetrics getMemoryMetrics();
    static ThroughputMetrics getThroughputMetrics();
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
#endif 
