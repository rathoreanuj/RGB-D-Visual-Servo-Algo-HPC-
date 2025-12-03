#include "ImageProcessor.h"
#include <algorithm>
#include <numeric>
#include <iostream>
cv::Mat ImageProcessor::convertToGrayscale(const cv::Mat& rgb) {
    cv::Mat grayscale;
    if (rgb.channels() == 3) {
        cv::cvtColor(rgb, grayscale, cv::COLOR_BGR2GRAY);
    } else {
        grayscale = rgb.clone();
    }
    return grayscale;
}
void ImageProcessor::computeHistogram(const cv::Mat& grayscale, std::vector<double>& histogram) {
    histogram.resize(256, 0.0);
    int totalPixels = grayscale.rows * grayscale.cols;
    for (int i = 0; i < grayscale.rows; ++i) {
        for (int j = 0; j < grayscale.cols; ++j) {
            int grayValue = static_cast<int>(grayscale.at<uchar>(i, j));
            histogram[grayValue] += 1.0;
        }
    }
    for (int i = 0; i < 256; ++i) {
        histogram[i] /= totalPixels;
    }
}
cv::Mat ImageProcessor::histogramEqualization(const cv::Mat& grayscale) {
    std::vector<double> histogram;
    computeHistogram(grayscale, histogram);
    std::vector<double> cdf(256, 0.0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
    std::vector<uchar> lookupTable(256);
    for (int i = 0; i < 256; ++i) {
        lookupTable[i] = static_cast<uchar>(std::round(cdf[i] * 255.0));
    }
    cv::Mat equalized = grayscale.clone();
    for (int i = 0; i < grayscale.rows; ++i) {
        for (int j = 0; j < grayscale.cols; ++j) {
            uchar originalValue = grayscale.at<uchar>(i, j);
            equalized.at<uchar>(i, j) = lookupTable[originalValue];
        }
    }
    return equalized;
}
cv::Mat ImageProcessor::segmentByThreshold(const cv::Mat& grayscale) {
    cv::Mat binary;
    cv::threshold(grayscale, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    return binary;
}
void ImageProcessor::extractTargetPoints(const cv::Mat& image, 
                                        std::vector<cv::Point2d>& points,
                                        bool useContours) {
    points.clear();
    cv::Mat binary;
    if (image.type() != CV_8UC1) {
        cv::cvtColor(image, binary, cv::COLOR_BGR2GRAY);
        cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);
    } else {
        binary = image.clone();
    }
    if (useContours) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (contours.empty()) {
            std::cerr << "Warning: No contours found in image" << std::endl;
            return;
        }
        size_t largestIdx = 0;
        double largestArea = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > largestArea) {
                largestArea = area;
                largestIdx = i;
            }
        }
        for (const auto& pt : contours[largestIdx]) {
            points.push_back(cv::Point2d(pt.x, pt.y));
        }
        std::cout << "Extracted " << points.size() << " contour points" << std::endl;
    } else {
        for (int i = 0; i < binary.rows; ++i) {
            for (int j = 0; j < binary.cols; ++j) {
                if (binary.at<uchar>(i, j) > 0) {
                    points.push_back(cv::Point2d(j, i));
                }
            }
        }
        std::cout << "Extracted " << points.size() << " foreground points" << std::endl;
    }
}
