#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "PoseEstimator.h"

/**
 * @brief Data logging utility for verification and plotting
 * 
 * Logs all outputs needed for the three verification plots:
 * 1. Histogram data (for Plot 1: Figure 6)
 * 2. Convergence data (for Plot 2: SA convergence)
 * 3. Final pose (for Plot 3: Pose visualization)
 */
class Logger {
public:
    /**
     * @brief Set output directory for all log files
     * @param outputDir Path to output directory
     */
    static void setOutputDirectory(const std::string& outputDir);
    
    /**
     * @brief Log histogram data for Plot 1
     * 
     * Saves original and equalized histograms to CSV:
     * Format: gray_value, original_count, equalized_count
     * 
     * @param originalHist Original histogram (256 values)
     * @param equalizedHist Equalized histogram (256 values)
     */
    static void logHistogramData(const std::vector<double>& originalHist,
                                 const std::vector<double>& equalizedHist);
    
    /**
     * @brief Log convergence data for Plot 2
     * 
     * Saves iteration number and best MHD score to CSV:
     * Format: iteration, mhd_score
     * 
     * @param iterationScores Vector of MHD scores (one per iteration)
     */
    static void logConvergenceData(const std::vector<double>& iterationScores);
    
    /**
     * @brief Log final pose for Plot 3
     * 
     * Saves final 6D pose parameters to JSON:
     * {
     *   "alpha": ..., "beta": ..., "gamma": ...,
     *   "tx": ..., "ty": ..., "tz": ...
     * }
     * 
     * @param pose Final optimized pose
     * @param finalScore Final MHD score
     */
    static void logFinalPose(const Pose6D& pose, double finalScore);
    
    /**
     * @brief Save image to output directory
     * @param image Image to save
     * @param filename Filename (e.g., "equalized.png")
     */
    static void saveImage(const cv::Mat& image, const std::string& filename);
    
    /**
     * @brief Log execution summary (console output)
     * @param totalIterations Total SA iterations
     * @param initialScore Initial MHD score
     * @param finalScore Final MHD score
     * @param convergenceRate (initialScore - finalScore) / initialScore
     */
    static void logSummary(int totalIterations, 
                          double initialScore,
                          double finalScore,
                          double convergenceRate);

private:
    static std::string outputDir_;
};

#endif // LOGGER_H
