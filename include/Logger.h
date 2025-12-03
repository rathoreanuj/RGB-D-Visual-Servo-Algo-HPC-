#ifndef LOGGER_H
#define LOGGER_H
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "PoseEstimator.h"
class Logger {
public:
    static void setOutputDirectory(const std::string& outputDir);
    static void logHistogramData(const std::vector<double>& originalHist,
                                 const std::vector<double>& equalizedHist);
    static void logConvergenceData(const std::vector<double>& iterationScores);
    static void logFinalPose(const Pose6D& pose, double finalScore);
    static void saveImage(const cv::Mat& image, const std::string& filename);
    static void logSummary(int totalIterations, 
                          double initialScore,
                          double finalScore,
                          double convergenceRate);
private:
    static std::string outputDir_;
};
#endif 
