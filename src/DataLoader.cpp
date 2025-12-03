#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
cv::Mat DataLoader::loadRGBImage(const std::string& filepath) {
    cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load RGB image: " + filepath);
    }
    std::cout << "Loaded RGB image: " << filepath 
              << " (Size: " << image.cols << "x" << image.rows << ")" << std::endl;
    return image;
}
cv::Mat DataLoader::loadDepthMap(const std::string& filepath) {
    cv::Mat depth;
    depth = cv::imread(filepath, cv::IMREAD_ANYDEPTH);
    if (!depth.empty()) {
        depth.convertTo(depth, CV_32F);
        std::cout << "Loaded depth map (PNG): " << filepath << std::endl;
        return depth;
    }
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to load depth map: " + filepath);
    }
    std::vector<std::vector<float>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',' || ss.peek() == ' ') {
                ss.ignore();
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    if (data.empty()) {
        throw std::runtime_error("Empty depth map file: " + filepath);
    }
    int rows = data.size();
    int cols = data[0].size();
    depth = cv::Mat(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            depth.at<float>(i, j) = data[i][j];
        }
    }
    std::cout << "Loaded depth map (TXT): " << filepath 
              << " (Size: " << cols << "x" << rows << ")" << std::endl;
    return depth;
}
WireframeModel DataLoader::loadOBJModel(const std::string& filepath) {
    WireframeModel model;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to load OBJ model: " + filepath);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            model.vertices.push_back(Eigen::Vector3d(x, y, z));
        }
        else if (type == "l") {
            int v1, v2;
            iss >> v1 >> v2;
            model.edges.push_back({v1 - 1, v2 - 1});  
        }
        else if (type == "f") {
            std::vector<int> faceVertices;
            int v;
            while (iss >> v) {
                faceVertices.push_back(v - 1);  
            }
            for (size_t i = 0; i < faceVertices.size(); ++i) {
                int v1 = faceVertices[i];
                int v2 = faceVertices[(i + 1) % faceVertices.size()];
                model.edges.push_back({v1, v2});
            }
        }
    }
    std::cout << "Loaded OBJ model: " << filepath 
              << " (Vertices: " << model.vertices.size() 
              << ", Edges: " << model.edges.size() << ")" << std::endl;
    if (model.vertices.empty()) {
        throw std::runtime_error("OBJ model has no vertices: " + filepath);
    }
    return model;
}
WireframeModel DataLoader::loadPLYModel(const std::string& filepath) {
    WireframeModel model;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to load PLY model: " + filepath);
    }
    std::string line;
    int numVertices = 0;
    int numEdges = 0;
    bool inHeader = true;
    while (std::getline(file, line) && inHeader) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;
        if (keyword == "element") {
            std::string type;
            int count;
            iss >> type >> count;
            if (type == "vertex") {
                numVertices = count;
            } else if (type == "edge") {
                numEdges = count;
            }
        } else if (keyword == "end_header") {
            inHeader = false;
        }
    }
    for (int i = 0; i < numVertices; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        double x, y, z;
        iss >> x >> y >> z;
        model.vertices.push_back(Eigen::Vector3d(x, y, z));
    }
    for (int i = 0; i < numEdges; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        int v1, v2;
        iss >> v1 >> v2;
        model.edges.push_back({v1, v2});
    }
    std::cout << "Loaded PLY model: " << filepath 
              << " (Vertices: " << model.vertices.size() 
              << ", Edges: " << model.edges.size() << ")" << std::endl;
    if (model.vertices.empty()) {
        throw std::runtime_error("PLY model has no vertices: " + filepath);
    }
    return model;
}
