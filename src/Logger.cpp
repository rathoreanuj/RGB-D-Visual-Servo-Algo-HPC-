#include "Logger.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
std::string Logger::outputDir_ = "output/";
void Logger::setOutputDirectory(const std::string& outputDir) {
    outputDir_ = outputDir;
    std::filesystem::create_directories(outputDir);
    std::filesystem::create_directories(outputDir + "/images");
    std::filesystem::create_directories(outputDir + "/data");
    std::filesystem::create_directories(outputDir + "/logs");
    std::cout << "Output directory set to: " << outputDir << std::endl;
}
void Logger::logHistogramData(const std::vector<double>& originalHist,
                              const std::vector<double>& equalizedHist) {
    std::string filepath = outputDir_ + "/data/histogram_data.csv";
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing" << std::endl;
        return;
    }
    file << "gray_value,original_count,equalized_count\n";
    for (size_t i = 0; i < 256; ++i) {
        file << i << ","
             << std::fixed << std::setprecision(6) << originalHist[i] << ","
             << equalizedHist[i] << "\n";
    }
    file.close();
    std::cout << "Histogram data saved to: " << filepath << std::endl;
}
void Logger::logConvergenceData(const std::vector<double>& iterationScores) {
    std::string filepath = outputDir_ + "/data/convergence_log.csv";
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing" << std::endl;
        return;
    }
    file << "iteration,mhd_score\n";
    for (size_t i = 0; i < iterationScores.size(); ++i) {
        file << i << ","
             << std::fixed << std::setprecision(6) << iterationScores[i] << "\n";
    }
    file.close();
    std::cout << "Convergence data saved to: " << filepath << std::endl;
}
void Logger::logFinalPose(const Pose6D& pose, double finalScore) {
    std::string filepath = outputDir_ + "/data/final_pose.json";
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing" << std::endl;
        return;
    }
    file << "{\n";
    file << "  \"pose\": {\n";
    file << "    \"alpha\": " << std::fixed << std::setprecision(6) << pose.alpha << ",\n";
    file << "    \"beta\": " << pose.beta << ",\n";
    file << "    \"gamma\": " << pose.gamma << ",\n";
    file << "    \"tx\": " << pose.tx << ",\n";
    file << "    \"ty\": " << pose.ty << ",\n";
    file << "    \"tz\": " << pose.tz << "\n";
    file << "  },\n";
    file << "  \"final_mhd_score\": " << finalScore << ",\n";
    file << "  \"units\": {\n";
    file << "    \"rotation\": \"radians\",\n";
    file << "    \"translation\": \"pixels_or_mm\"\n";
    file << "  }\n";
    file << "}\n";
    file.close();
    std::cout << "Final pose saved to: " << filepath << std::endl;
}
void Logger::saveImage(const cv::Mat& image, const std::string& filename) {
    std::string filepath = outputDir_ + "/images/" + filename;
    if (cv::imwrite(filepath, image)) {
        std::cout << "Image saved to: " << filepath << std::endl;
    } else {
        std::cerr << "Error: Could not save image to " << filepath << std::endl;
    }
}
void Logger::logSummary(int totalIterations, 
                       double initialScore,
                       double finalScore,
                       double convergenceRate) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "         OPTIMIZATION SUMMARY          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total iterations:    " << totalIterations << std::endl;
    std::cout << "Initial MHD score:   " << std::fixed << std::setprecision(4) << initialScore << std::endl;
    std::cout << "Final MHD score:     " << finalScore << std::endl;
    std::cout << "Improvement:         " << (initialScore - finalScore) << std::endl;
    std::cout << "Convergence rate:    " << std::setprecision(2) << (convergenceRate * 100) << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;
}
