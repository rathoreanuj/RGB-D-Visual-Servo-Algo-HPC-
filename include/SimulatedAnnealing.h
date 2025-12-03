#ifndef SIMULATEDANNEALING_H
#define SIMULATEDANNEALING_H
#include <vector>
#include <functional>
#include <random>
#include "PoseEstimator.h"
class SimulatedAnnealing {
public:
    struct SAConfig {
        double initialTemperature = 100.0;  
        double coolingRate = 0.95;          
        int maxIterations = 500;
        double minTemperature = 0.1;
        double alpha_min = -M_PI, alpha_max = M_PI;      
        double beta_min = -M_PI, beta_max = M_PI;        
        double gamma_min = -M_PI, gamma_max = M_PI;      
        double tx_min = -500.0, tx_max = 500.0;          
        double ty_min = -500.0, ty_max = 500.0;          
        double tz_min = 100.0, tz_max = 2000.0;          
        double stepSize_rotation = 0.1;     
        double stepSize_translation = 10.0; 
        unsigned int randomSeed = 42;  
    };
    using ObjectiveFunction = std::function<double(const Pose6D&)>;
    explicit SimulatedAnnealing(const SAConfig& config);
    Pose6D optimize(ObjectiveFunction objectiveFunc,
                   const Pose6D& initialPose,
                   std::vector<double>& iterationLog);
private:
    Pose6D generateNeighbor(const Pose6D& current);
    bool acceptanceDecision(double currentCost, double newCost, double temperature);
    double updateTemperature(int iteration);
    SAConfig config_;
    std::mt19937 rng_;  
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
};
#endif 
