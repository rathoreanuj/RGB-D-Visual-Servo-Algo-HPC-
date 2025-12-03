#ifndef MHD_H
#define MHD_H
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
class MHD {
public:
    static double pointToLineSegmentDistance(const cv::Point2d& point,
                                            const cv::Point2d& lineStart,
                                            const cv::Point2d& lineEnd);
    static double directedHausdorffDistance(
        const std::vector<cv::Point2d>& pointSet,
        const std::vector<std::pair<cv::Point2d, cv::Point2d>>& lineSegments);
    static double computeMHD(
        const std::vector<cv::Point2d>& targetPoints,
        const std::vector<std::pair<cv::Point2d, cv::Point2d>>& modelEdges);
};
#endif 
