#ifndef DATALOADER_H
#define DATALOADER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @brief Structure to hold a 3D wireframe model
 * 
 * The model consists of vertices (3D points) and edges (line segments).
 * This represents the set M from the paper (Section 3.3).
 */
struct WireframeModel {
    std::vector<Eigen::Vector3d> vertices;  // 3D vertices
    std::vector<std::pair<int, int>> edges; // Edge indices (pairs of vertex indices)
};

/**
 * @brief Data loading utility class
 * 
 * Loads RGB images, depth maps, and 3D wireframe models from disk.
 */
class DataLoader {
public:
    /**
     * @brief Load RGB image from file
     * @param filepath Path to RGB image (PNG/JPG)
     * @return cv::Mat RGB image
     */
    static cv::Mat loadRGBImage(const std::string& filepath);
    
    /**
     * @brief Load depth map from file
     * @param filepath Path to depth map (PNG/TXT)
     * @return cv::Mat Depth map (single channel, float)
     */
    static cv::Mat loadDepthMap(const std::string& filepath);
    
    /**
     * @brief Load 3D wireframe model from OBJ file
     * @param filepath Path to .obj file
     * @return WireframeModel Parsed 3D model
     */
    static WireframeModel loadOBJModel(const std::string& filepath);
    
    /**
     * @brief Load 3D wireframe model from PLY file
     * @param filepath Path to .ply file
     * @return WireframeModel Parsed 3D model
     */
    static WireframeModel loadPLYModel(const std::string& filepath);
};

#endif // DATALOADER_H
