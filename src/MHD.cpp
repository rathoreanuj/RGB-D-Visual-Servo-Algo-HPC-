#include "MHD.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <cfloat>
double MHD::pointToLineSegmentDistance(const cv::Point2d& point,
                                       const cv::Point2d& lineStart,
                                       const cv::Point2d& lineEnd) {
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    double lengthSq = dx * dx + dy * dy;
    if (lengthSq < 1e-10) {
        double distX = point.x - lineStart.x;
        double distY = point.y - lineStart.y;
        return std::sqrt(distX * distX + distY * distY);
    }
    double t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lengthSq;
    t = std::max(0.0, std::min(1.0, t));
    double closestX = lineStart.x + t * dx;
    double closestY = lineStart.y + t * dy;
    double distX = point.x - closestX;
    double distY = point.y - closestY;
    return std::sqrt(distX * distX + distY * distY);
}
double MHD::directedHausdorffDistance(
    const std::vector<cv::Point2d>& pointSet,
    const std::vector<std::pair<cv::Point2d, cv::Point2d>>& lineSegments) {
    if (pointSet.empty() || lineSegments.empty()) {
        return std::numeric_limits<double>::max();
    }
    double maxMinDistance = 0.0;
    #pragma omp parallel for reduction(max:maxMinDistance) schedule(guided, 32) if(pointSet.size() > 500)
    for (int i = 0; i < static_cast<int>(pointSet.size()); i++) {
        const auto& point = pointSet[i];
        double minDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < static_cast<int>(lineSegments.size()); j++) {
            double dist = pointToLineSegmentDistance(point, lineSegments[j].first, lineSegments[j].second);
            if (dist < minDistance) {
                minDistance = dist;
            }
        }
        if (minDistance > maxMinDistance) {
            maxMinDistance = minDistance;
        }
    }
    return maxMinDistance;
}
double MHD::computeMHD(
    const std::vector<cv::Point2d>& targetPoints,
    const std::vector<std::pair<cv::Point2d, cv::Point2d>>& modelEdges) {
    if (targetPoints.empty() || modelEdges.empty()) {
        return std::numeric_limits<double>::max();
    }
    std::vector<cv::Point2d> modelPoints;
    for (const auto& edge : modelEdges) {
        modelPoints.push_back(edge.first);
        modelPoints.push_back(edge.second);
    }
    double h_T_M = directedHausdorffDistance(targetPoints, modelEdges);
    double h_M_T = 0.0;
    #pragma omp parallel for reduction(max:h_M_T) schedule(guided, 32) if(modelPoints.size() > 500)
    for (int i = 0; i < static_cast<int>(modelPoints.size()); i++) {
        const auto& modelPt = modelPoints[i];
        double minDist = std::numeric_limits<double>::max();
        for (int j = 0; j < static_cast<int>(targetPoints.size()); j++) {
            double dx = modelPt.x - targetPoints[j].x;
            double dy = modelPt.y - targetPoints[j].y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
            }
        }
        if (minDist > h_M_T) {
            h_M_T = minDist;
        }
    }
    double mhd = std::max(h_T_M, h_M_T);
    return mhd;
}
