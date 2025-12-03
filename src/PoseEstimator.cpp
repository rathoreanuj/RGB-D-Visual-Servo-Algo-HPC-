#include "PoseEstimator.h"
#include <cmath>
#include <omp.h>
void PoseEstimator::setCameraIntrinsics(double fx, double fy, double cx, double cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
}
Eigen::Matrix3d PoseEstimator::computeRotationMatrix(double alpha, double beta, double gamma) {
    Eigen::Matrix3d Rx;
    Rx << 1,             0,              0,
          0,  std::cos(alpha), -std::sin(alpha),
          0,  std::sin(alpha),  std::cos(alpha);
    Eigen::Matrix3d Ry;
    Ry <<  std::cos(beta), 0, std::sin(beta),
                        0, 1,             0,
          -std::sin(beta), 0, std::cos(beta);
    Eigen::Matrix3d Rz;
    Rz << std::cos(gamma), -std::sin(gamma), 0,
          std::sin(gamma),  std::cos(gamma), 0,
                        0,                0, 1;
    return Rz * Ry * Rx;
}
Eigen::Matrix4d PoseEstimator::computeTransformationMatrix(const Pose6D& pose) {
    Eigen::Matrix3d R = computeRotationMatrix(pose.alpha, pose.beta, pose.gamma);
    Eigen::Vector3d t(pose.tx, pose.ty, pose.tz);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M.block<3, 3>(0, 0) = R;
    M.block<3, 1>(0, 3) = t;
    return M;
}
cv::Point2d PoseEstimator::project3Dto2D(const Eigen::Vector3d& point3D) const {
    if (std::abs(point3D.z()) < 1e-6) {
        return cv::Point2d(0, 0);
    }
    double u = fx_ * (point3D.x() / point3D.z()) + cx_;
    double v = fy_ * (point3D.y() / point3D.z()) + cy_;
    return cv::Point2d(u, v);
}
std::vector<std::pair<cv::Point2d, cv::Point2d>> PoseEstimator::projectModel(
    const WireframeModel& model,
    const Pose6D& pose) const {
    Eigen::Matrix4d T = computeTransformationMatrix(pose);
    std::vector<Eigen::Vector3d> transformedVertices;
    transformedVertices.reserve(model.vertices.size());
    transformedVertices.resize(model.vertices.size());
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < static_cast<int>(model.vertices.size()); i++) {
        const auto& vertex = model.vertices[i];
        Eigen::Vector4d vertexHomogeneous(vertex.x(), vertex.y(), vertex.z(), 1.0);
        Eigen::Vector4d transformedHomogeneous = T * vertexHomogeneous;
        transformedVertices[i] = Eigen::Vector3d(
            transformedHomogeneous.x(),
            transformedHomogeneous.y(),
            transformedHomogeneous.z()
        );
    }
    std::vector<std::pair<cv::Point2d, cv::Point2d>> projectedEdges;
    int maxThreads = omp_get_max_threads();
    std::vector<std::vector<std::pair<cv::Point2d, cv::Point2d>>> threadResults(maxThreads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        threadResults[tid].reserve(model.edges.size() / maxThreads + 10);
        #pragma omp for schedule(dynamic, 4)
        for (int e = 0; e < static_cast<int>(model.edges.size()); e++) {
            const auto& edge = model.edges[e];
            int v1_idx = edge.first;
            int v2_idx = edge.second;
            if (v1_idx >= 0 && v1_idx < static_cast<int>(transformedVertices.size()) &&
                v2_idx >= 0 && v2_idx < static_cast<int>(transformedVertices.size())) {
                const Eigen::Vector3d& v1_3d = transformedVertices[v1_idx];
                const Eigen::Vector3d& v2_3d = transformedVertices[v2_idx];
                if (v1_3d.z() > 0 && v2_3d.z() > 0) {
                    cv::Point2d v1_2d = project3Dto2D(v1_3d);
                    cv::Point2d v2_2d = project3Dto2D(v2_3d);
                    threadResults[tid].push_back({v1_2d, v2_2d});
                }
            }
        }
    }
    size_t totalSize = 0;
    for (const auto& results : threadResults) {
        totalSize += results.size();
    }
    projectedEdges.reserve(totalSize);
    for (const auto& results : threadResults) {
        projectedEdges.insert(projectedEdges.end(), results.begin(), results.end());
    }
    return projectedEdges;
}
