#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include "DataLoader.h"
struct Pose6D {
    double alpha;  
    double beta;   
    double gamma;  
    double tx;     
    double ty;     
    double tz;     
    Pose6D() : alpha(0), beta(0), gamma(0), tx(0), ty(0), tz(0) {}
    Pose6D(double a, double b, double g, double x, double y, double z)
        : alpha(a), beta(b), gamma(g), tx(x), ty(y), tz(z) {}
};
class PoseEstimator {
public:
    void setCameraIntrinsics(double fx, double fy, double cx, double cy);
    static Eigen::Matrix3d computeRotationMatrix(double alpha, double beta, double gamma);
    static Eigen::Matrix4d computeTransformationMatrix(const Pose6D& pose);
    cv::Point2d project3Dto2D(const Eigen::Vector3d& point3D) const;
    std::vector<std::pair<cv::Point2d, cv::Point2d>> projectModel(
        const WireframeModel& model,
        const Pose6D& pose) const;
private:
    double fx_, fy_, cx_, cy_;  
};
#endif 
