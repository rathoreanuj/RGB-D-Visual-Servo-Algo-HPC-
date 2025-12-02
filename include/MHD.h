#ifndef MHD_H
#define MHD_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

/**
 * @brief Modified Hausdorff Distance (MHD) calculator
 * 
 * Implements Section 3.3 (Equations 9-11) from the paper.
 * Measures the matching quality between a 2D point set T (from image)
 * and a projected 3D model M^.
 */
class MHD {
public:
    /**
     * @brief Calculate point-to-line-segment distance
     * 
     * Implements Equation 11 from paper (Figure 7).
     * Computes the perpendicular distance from point t_j to line segment m_i.
     * 
     * @param point 2D point t_j
     * @param lineStart Start point of line segment
     * @param lineEnd End point of line segment
     * @return double Distance from point to line segment
     */
    static double pointToLineSegmentDistance(const cv::Point2d& point,
                                            const cv::Point2d& lineStart,
                                            const cv::Point2d& lineEnd);
    
    /**
     * @brief Calculate directed Hausdorff distance h(A, B)
     * 
     * h(A, B) = max_{a in A} min_{b in B} d(a, b)
     * 
     * For point set to line segments:
     * h(T, M^) = max_{t in T} min_{m in M^} d(t, m)
     * 
     * @param pointSet Point set (e.g., T from image)
     * @param lineSegments Line segments (e.g., M^ projected model edges)
     * @return double Directed Hausdorff distance
     */
    static double directedHausdorffDistance(
        const std::vector<cv::Point2d>& pointSet,
        const std::vector<std::pair<cv::Point2d, cv::Point2d>>& lineSegments);
    
    /**
     * @brief Calculate Modified Hausdorff Distance (MHD)
     * 
     * Implements Equation 10 from paper:
     * H(M^, T) = max(h(M^, T), h(T, M^))
     * 
     * This is the bidirectional Hausdorff distance.
     * 
     * @param targetPoints Point set T extracted from image
     * @param modelEdges Projected model edges M^ (2D line segments)
     * @return double MHD score (lower is better)
     */
    static double computeMHD(
        const std::vector<cv::Point2d>& targetPoints,
        const std::vector<std::pair<cv::Point2d, cv::Point2d>>& modelEdges);
};

#endif // MHD_H
