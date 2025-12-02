#include "SimulatedAnnealing.h"
#include <cmath>
#include <iostream>
#include <omp.h>

SimulatedAnnealing::SimulatedAnnealing(const SAConfig& config) 
    : config_(config) {
    // CRITICAL: Seed RNG with fixed value for determinism
    rng_.seed(config_.randomSeed);
    std::cout << "Simulated Annealing initialized with seed: " << config_.randomSeed << std::endl;
}

Pose6D SimulatedAnnealing::generateNeighbor(const Pose6D& current) {
    // Generate neighbor by adding small random perturbations
    Pose6D neighbor = current;
    
    // Use normal distribution for perturbations (more realistic than uniform)
    std::normal_distribution<double> rotation_dist(0.0, config_.stepSize_rotation);
    std::normal_distribution<double> translation_dist(0.0, config_.stepSize_translation);
    
    // Perturb rotation angles
    neighbor.alpha += rotation_dist(rng_);
    neighbor.beta += rotation_dist(rng_);
    neighbor.gamma += rotation_dist(rng_);
    
    // Perturb translation
    neighbor.tx += translation_dist(rng_);
    neighbor.ty += translation_dist(rng_);
    neighbor.tz += translation_dist(rng_);
    
    // Clamp to search space bounds
    neighbor.alpha = std::max(config_.alpha_min, std::min(config_.alpha_max, neighbor.alpha));
    neighbor.beta = std::max(config_.beta_min, std::min(config_.beta_max, neighbor.beta));
    neighbor.gamma = std::max(config_.gamma_min, std::min(config_.gamma_max, neighbor.gamma));
    neighbor.tx = std::max(config_.tx_min, std::min(config_.tx_max, neighbor.tx));
    neighbor.ty = std::max(config_.ty_min, std::min(config_.ty_max, neighbor.ty));
    neighbor.tz = std::max(config_.tz_min, std::min(config_.tz_max, neighbor.tz));
    
    return neighbor;
}

bool SimulatedAnnealing::acceptanceDecision(double currentCost, double newCost, double temperature) {
    // Metropolis acceptance criterion (Boltzmann distribution)
    
    // Always accept if new solution is better
    if (newCost < currentCost) {
        return true;
    }
    
    // Accept worse solution with probability exp(-ΔE/T)
    double deltaE = newCost - currentCost;
    double acceptanceProbability = std::exp(-deltaE / temperature);
    
    double randomValue = uniform_dist_(rng_);
    return randomValue < acceptanceProbability;
}

double SimulatedAnnealing::updateTemperature(int iteration) {
    // Geometric cooling schedule: T_k = T_0 * α^k
    return config_.initialTemperature * std::pow(config_.coolingRate, iteration);
}

Pose6D SimulatedAnnealing::optimize(ObjectiveFunction objectiveFunc,
                                   const Pose6D& initialPose,
                                   std::vector<double>& iterationLog) {
    // Implements SERIAL Simulated Annealing
    // Parallelization happens in the objective function (MHD computation)
    
    iterationLog.clear();
    
    std::cout << "\n=== Starting Simulated Annealing ===" << std::endl;
    std::cout << "Initial temperature: " << config_.initialTemperature << std::endl;
    std::cout << "Max iterations: " << config_.maxIterations << std::endl;
    std::cout << "OpenMP threads available: " << omp_get_max_threads() << std::endl;
    
    // Initialize current state
    Pose6D currentPose = initialPose;
    double currentCost = objectiveFunc(currentPose);
    
    // Track best solution
    Pose6D bestPose = currentPose;
    double bestCost = currentCost;
    
    std::cout << "Initial cost: " << currentCost << std::endl;
    
    // Main SA loop
    for (int iter = 0; iter < config_.maxIterations; ++iter) {
        // Update temperature (geometric cooling)
        double temperature = config_.initialTemperature * std::pow(config_.coolingRate, iter);
        
        // Check termination condition
        if (temperature < config_.minTemperature) {
            std::cout << "Terminated early at iteration " << iter << " (T < T_min)" << std::endl;
            break;
        }
        
        // Generate neighbor state
        Pose6D neighborPose = generateNeighbor(currentPose);
        
        // Evaluate objective function (THIS IS WHERE PARALLELISM HAPPENS via OpenMP in MHD)
        double neighborCost = objectiveFunc(neighborPose);
        
        // Acceptance decision (Metropolis criterion)
        bool accept = acceptanceDecision(currentCost, neighborCost, temperature);
        
        if (accept) {
            currentPose = neighborPose;
            currentCost = neighborCost;
            
            // Update global best if improved
            if (currentCost < bestCost) {
                bestPose = currentPose;
                bestCost = currentCost;
            }
        }
        
        // Log best cost at this iteration
        iterationLog.push_back(bestCost);
        
        // Print progress every 50 iterations
        if (iter % 50 == 0 || iter == config_.maxIterations - 1) {
            std::cout << "Iter " << iter 
                      << " | Temp: " << temperature
                      << " | Current: " << currentCost
                      << " | Best: " << bestCost << std::endl;
        }
    }
    
    std::cout << "\n=== Optimization Complete ===" << std::endl;
    std::cout << "Final best cost: " << bestCost << std::endl;
    std::cout << "Total iterations: " << iterationLog.size() << std::endl;
    
    return bestPose;
}
