#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include "DataLoader.h"

/**
 * @brief 6D pose representation
 * 
 * Represents the 6 degrees of freedom: (α, β, γ, tx, ty, tz)
 * - (α, β, γ): Rotation angles (Euler angles in radians)
 * - (tx, ty, tz): Translation vector
 */
struct Pose6D {
    double alpha;  // Rotation around X-axis (radians)
    double beta;   // Rotation around Y-axis (radians)
    double gamma;  // Rotation around Z-axis (radians)
    double tx;     // Translation X (mm or pixels)
    double ty;     // Translation Y
    double tz;     // Translation Z (depth)
    
    Pose6D() : alpha(0), beta(0), gamma(0), tx(0), ty(0), tz(0) {}
    
    Pose6D(double a, double b, double g, double x, double y, double z)
        : alpha(a), beta(b), gamma(g), tx(x), ty(y), tz(z) {}
};

/**
 * @brief 3D-to-2D projection and pose transformation utilities
 * 
 * Implements Equations 14-17 from the paper.
 */
class PoseEstimator {
public:
    /**
     * @brief Set camera intrinsic parameters
     * 
     * Camera calibration matrix K:
     * [fx  0  cx]
     * [0  fy  cy]
     * [0   0   1]
     * 
     * @param fx Focal length X (pixels)
     * @param fy Focal length Y (pixels)
     * @param cx Principal point X (pixels)
     * @param cy Principal point Y (pixels)
     */
    void setCameraIntrinsics(double fx, double fy, double cx, double cy);
    
    /**
     * @brief Compute rotation matrix from Euler angles
     * 
     * Implements Equation 17 from paper:
     * R(α, β, γ) = Rz(γ) * Ry(β) * Rx(α)
     * 
     * @param alpha Rotation around X-axis (radians)
     * @param beta Rotation around Y-axis (radians)
     * @param gamma Rotation around Z-axis (radians)
     * @return Eigen::Matrix3d 3x3 rotation matrix
     */
    static Eigen::Matrix3d computeRotationMatrix(double alpha, double beta, double gamma);
    
    /**
     * @brief Compute transformation matrix
     * 
     * Implements Equation 14 from paper:
     * M = T(tx, ty, tz) · R(α, β, γ)
     * 
     * Returns 4x4 homogeneous transformation matrix.
     * 
     * @param pose 6D pose parameters
     * @return Eigen::Matrix4d 4x4 transformation matrix
     */
    static Eigen::Matrix4d computeTransformationMatrix(const Pose6D& pose);
    
    /**
     * @brief Project 3D point to 2D image plane
     * 
     * Uses camera intrinsics to project 3D point (X, Y, Z) to 2D pixel (u, v).
     * 
     * @param point3D 3D point in camera frame
     * @return cv::Point2d 2D pixel coordinates
     */
    cv::Point2d project3Dto2D(const Eigen::Vector3d& point3D) const;
    
    /**
     * @brief Project 3D wireframe model to 2D image plane
     * 
     * Applies 6D pose transformation and camera projection to generate
     * 2D line segments M^ (Section 3.3).
     * 
     * @param model 3D wireframe model M
     * @param pose 6D pose (α, β, γ, tx, ty, tz)
     * @return std::vector of 2D line segments (projected edges)
     */
    std::vector<std::pair<cv::Point2d, cv::Point2d>> projectModel(
        const WireframeModel& model,
        const Pose6D& pose) const;

private:
    double fx_, fy_, cx_, cy_;  // Camera intrinsic parameters
};

#endif // POSEESTIMATOR_H
