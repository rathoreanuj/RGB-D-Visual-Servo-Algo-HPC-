#include "MHD.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <cfloat>

double MHD::pointToLineSegmentDistance(const cv::Point2d& point,
                                       const cv::Point2d& lineStart,
                                       const cv::Point2d& lineEnd) {
    // Implements Equation 11 and Figure 7 from the paper
    // Calculate perpendicular distance from point to line segment
    
    // Vector from lineStart to lineEnd
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    
    // Length squared of line segment
    double lengthSq = dx * dx + dy * dy;
    
    if (lengthSq < 1e-10) {
        // Degenerate case: line segment is a point
        double distX = point.x - lineStart.x;
        double distY = point.y - lineStart.y;
        return std::sqrt(distX * distX + distY * distY);
    }
    
    // Calculate parameter t for projection onto line
    // t = ((point - lineStart) · (lineEnd - lineStart)) / |lineEnd - lineStart|^2
    double t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lengthSq;
    
    // Clamp t to [0, 1] to stay on line segment
    t = std::max(0.0, std::min(1.0, t));
    
    // Find closest point on line segment
    double closestX = lineStart.x + t * dx;
    double closestY = lineStart.y + t * dy;
    
    // Calculate distance from point to closest point
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
    
    // Directed Hausdorff distance: h(A, B) = max_{a in A} min_{b in B} d(a, b)
    // For point set to line segments: h(T, M^) = max_{t in T} min_{m in M^} d(t, m)
    
    double maxMinDistance = 0.0;
    
    // OPENMP PARALLELIZATION: Parallel loop over target points
    // Use guided scheduling for better load balancing
    #pragma omp parallel for reduction(max:maxMinDistance) schedule(guided, 32) if(pointSet.size() > 500)
    for (int i = 0; i < static_cast<int>(pointSet.size()); i++) {
        const auto& point = pointSet[i];
        
        // Find minimum distance from this point to any line segment
        double minDistance = std::numeric_limits<double>::max();
        
        // Inner loop: sequential (too small for SIMD overhead)
        for (int j = 0; j < static_cast<int>(lineSegments.size()); j++) {
            double dist = pointToLineSegmentDistance(point, lineSegments[j].first, lineSegments[j].second);
            if (dist < minDistance) {
                minDistance = dist;
            }
        }
        
        // Track the maximum of these minimum distances
        if (minDistance > maxMinDistance) {
            maxMinDistance = minDistance;
        }
    }
    
    return maxMinDistance;
}

double MHD::computeMHD(
    const std::vector<cv::Point2d>& targetPoints,
    const std::vector<std::pair<cv::Point2d, cv::Point2d>>& modelEdges) {
    
    // Implements Equation 10 from the paper:
    // H(M^, T) = max(h(M^, T), h(T, M^))
    // This is the bidirectional Modified Hausdorff Distance
    
    if (targetPoints.empty() || modelEdges.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    // Convert model edges to point set for reverse direction
    std::vector<cv::Point2d> modelPoints;
    for (const auto& edge : modelEdges) {
        modelPoints.push_back(edge.first);
        modelPoints.push_back(edge.second);
    }
    
    // Calculate h(T, M^) - distance from target points to model edges
    double h_T_M = directedHausdorffDistance(targetPoints, modelEdges);
    
    // Calculate h(M^, T) - distance from model points to target
    // For this direction, we need to treat target points as line segments too
    // However, for simplicity and following the paper's approach,
    // we compute point-to-point distances instead
    double h_M_T = 0.0;
    
    // OPENMP PARALLELIZATION: Parallel loop over model points
    #pragma omp parallel for reduction(max:h_M_T) schedule(guided, 32) if(modelPoints.size() > 500)
    for (int i = 0; i < static_cast<int>(modelPoints.size()); i++) {
        const auto& modelPt = modelPoints[i];
        double minDist = std::numeric_limits<double>::max();
        
        // Inner loop: sequential (too small for nested parallelism)
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
    
    // Bidirectional Hausdorff distance (Equation 10)
    double mhd = std::max(h_T_M, h_M_T);
    
    return mhd;
}
