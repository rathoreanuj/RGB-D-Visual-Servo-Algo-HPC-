#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Image preprocessing and segmentation utilities
 * 
 * Implements histogram equalization (Section 3.2, Equations 6-8)
 * and basic image segmentation.
 */
class ImageProcessor {
public:
    /**
     * @brief Convert RGB image to grayscale
     * @param rgb RGB input image
     * @return cv::Mat Grayscale image
     */
    static cv::Mat convertToGrayscale(const cv::Mat& rgb);
    
    /**
     * @brief Compute histogram of grayscale image
     * 
     * Implements Equation 6 from paper:
     * p(r_i) = (Number of pixels with gray value r_i) / (Total number of pixels)
     * 
     * @param grayscale Input grayscale image
     * @param histogram Output histogram (256 bins)
     */
    static void computeHistogram(const cv::Mat& grayscale, std::vector<double>& histogram);
    
    /**
     * @brief Perform histogram equalization
     * 
     * Implements Equations 7-8 from paper:
     * Eq. 7: s_k = T(r_k) = sum_{j=0}^{k} p(r_j)
     * Eq. 8: Maps original pixel values to equalized values
     * 
     * @param grayscale Input grayscale image
     * @return cv::Mat Histogram-equalized image
     */
    static cv::Mat histogramEqualization(const cv::Mat& grayscale);
    
    /**
     * @brief Extract target point set T from image
     * 
     * Uses simple thresholding or contour detection to extract
     * the boundary points of the target object (Section 3.3).
     * 
     * @param image Input image (grayscale or binary)
     * @param points Output point set T (2D pixel coordinates)
     * @param useContours If true, extract contour points; otherwise use all foreground points
     */
    static void extractTargetPoints(const cv::Mat& image, 
                                    std::vector<cv::Point2d>& points,
                                    bool useContours = true);
    
    /**
     * @brief Simple binary segmentation using Otsu's thresholding
     * @param grayscale Input grayscale image
     * @return cv::Mat Binary mask (0 = background, 255 = foreground)
     */
    static cv::Mat segmentByThreshold(const cv::Mat& grayscale);
};

#endif // IMAGEPROCESSOR_H
