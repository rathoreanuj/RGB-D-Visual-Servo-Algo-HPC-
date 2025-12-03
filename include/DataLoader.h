#ifndef DATALOADER_H
#define DATALOADER_H
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
struct WireframeModel {
    std::vector<Eigen::Vector3d> vertices;  
    std::vector<std::pair<int, int>> edges; 
};
class DataLoader {
public:
    static cv::Mat loadRGBImage(const std::string& filepath);
    static cv::Mat loadDepthMap(const std::string& filepath);
    static WireframeModel loadOBJModel(const std::string& filepath);
    static WireframeModel loadPLYModel(const std::string& filepath);
};
#endif 
