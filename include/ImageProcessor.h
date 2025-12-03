#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>
#include <vector>
class ImageProcessor {
public:
    static cv::Mat convertToGrayscale(const cv::Mat& rgb);
    static void computeHistogram(const cv::Mat& grayscale, std::vector<double>& histogram);
    static cv::Mat histogramEqualization(const cv::Mat& grayscale);
    static void extractTargetPoints(const cv::Mat& image, 
                                    std::vector<cv::Point2d>& points,
                                    bool useContours = true);
    static cv::Mat segmentByThreshold(const cv::Mat& grayscale);
};
#endif 
