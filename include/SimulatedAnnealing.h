#ifndef SIMULATEDANNEALING_H
#define SIMULATEDANNEALING_H

#include <vector>
#include <functional>
#include <random>
#include "PoseEstimator.h"

/**
 * @brief Simulated Annealing optimizer for 6D pose estimation
 * 
 * Implements the SA algorithm from Figure 8 of the paper.
 * Minimizes the MHD objective function to find optimal pose.
 * 
 * CRITICAL: Uses fixed random seed for deterministic behavior.
 */
class SimulatedAnnealing {
public:
    /**
     * @brief Configuration parameters for SA algorithm
     */
    struct SAConfig {
        double initialTemperature = 100.0;  // T_0
        double coolingRate = 0.95;          // α (T_k = T_0 * α^k)
        int maxIterations = 500;
        double minTemperature = 0.1;
        
        // Search space bounds for 6D pose
        double alpha_min = -M_PI, alpha_max = M_PI;      // Rotation X (radians)
        double beta_min = -M_PI, beta_max = M_PI;        // Rotation Y
        double gamma_min = -M_PI, gamma_max = M_PI;      // Rotation Z
        double tx_min = -500.0, tx_max = 500.0;          // Translation X (pixels)
        double ty_min = -500.0, ty_max = 500.0;          // Translation Y
        double tz_min = 100.0, tz_max = 2000.0;          // Translation Z (mm)
        
        // Step sizes for neighbor generation
        double stepSize_rotation = 0.1;     // Radians
        double stepSize_translation = 10.0; // Pixels or mm
        
        // Determinism
        unsigned int randomSeed = 42;  // Fixed seed for reproducibility
    };
    
    /**
     * @brief Objective function type
     * 
     * Takes a Pose6D and returns a scalar cost (MHD score).
     * Lower is better.
     */
    using ObjectiveFunction = std::function<double(const Pose6D&)>;
    
    /**
     * @brief Constructor
     * @param config SA configuration parameters
     */
    explicit SimulatedAnnealing(const SAConfig& config);
    
    /**
     * @brief Optimize 6D pose using Simulated Annealing
     * 
     * Implements the algorithm flowchart from Figure 8:
     * 1. Initialize with random or given initial pose
     * 2. Generate neighbor state (perturb current pose)
     * 3. Evaluate objective function (MHD)
     * 4. Accept/reject based on Metropolis criterion
     * 5. Update temperature
     * 6. Log best score at each iteration
     * 7. Repeat until convergence or max iterations
     * 
     * @param objectiveFunc Objective function to minimize (MHD)
     * @param initialPose Initial guess for pose
     * @param iterationLog Output vector to log best score at each iteration
     * @return Pose6D Optimized pose
     */
    Pose6D optimize(ObjectiveFunction objectiveFunc,
                   const Pose6D& initialPose,
                   std::vector<double>& iterationLog);

private:
    /**
     * @brief Generate a neighbor state by perturbing current pose
     * 
     * Adds small random perturbations to pose parameters.
     * 
     * @param current Current pose
     * @return Pose6D Neighboring pose
     */
    Pose6D generateNeighbor(const Pose6D& current);
    
    /**
     * @brief Metropolis acceptance criterion
     * 
     * Accept new state if:
     * - Cost decreased (ΔE < 0), OR
     * - With probability exp(-ΔE/T) (Boltzmann distribution)
     * 
     * @param currentCost Cost of current state
     * @param newCost Cost of new state
     * @param temperature Current temperature
     * @return bool True if new state should be accepted
     */
    bool acceptanceDecision(double currentCost, double newCost, double temperature);
    
    /**
     * @brief Update temperature according to cooling schedule
     * 
     * T_k = T_0 * α^k (geometric cooling)
     * 
     * @param iteration Current iteration number
     * @return double New temperature
     */
    double updateTemperature(int iteration);
    
    SAConfig config_;
    std::mt19937 rng_;  // Mersenne Twister RNG (seeded for determinism)
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
};

#endif // SIMULATEDANNEALING_H
