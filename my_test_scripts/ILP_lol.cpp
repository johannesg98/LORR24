#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "ortools/linear_solver/linear_solver.h"

using namespace operations_research;

int main() {
    // Initialize random seed
    std::srand(std::time(nullptr));

    // Problem size
    const int num_agents = 600;
    const int num_tasks = 300;

    // Generate a random cost matrix
    std::vector<std::vector<int>> cost_matrix(num_agents, std::vector<int>(num_tasks));
    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_tasks; ++j) {
            cost_matrix[i][j] = 2 + (std::rand() % 99);  // Random cost in [2, 100]
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Create the solver
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("CBC"));
    if (!solver) {
        std::cerr << "CBC solver not available." << std::endl;
        return 1;
    }

    // Decision variables: x[i][j] = 1 if agent i is assigned to task j, else 0
    std::vector<std::vector<const MPVariable*>> x(num_agents, std::vector<const MPVariable*>(num_tasks, nullptr));
    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_tasks; ++j) {
            x[i][j] = solver->MakeIntVar(0, 1, "x_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // Objective function: Minimize the total cost
    MPObjective* const objective = solver->MutableObjective();
    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_tasks; ++j) {
            objective->SetCoefficient(x[i][j], cost_matrix[i][j]);
        }
    }
    objective->SetMinimization();

    // Constraints: Each agent is assigned to at most one task
    for (int i = 0; i < num_agents; ++i) {
        LinearExpr agent_constraint;
        for (int j = 0; j < num_tasks; ++j) {
            agent_constraint += x[i][j];
        }
        solver->MakeRowConstraint(agent_constraint <= 1, "agent_" + std::to_string(i) + "_at_most_one_task");
    }

    // Constraints: Each task is assigned to exactly one agent
    for (int j = 0; j < num_tasks; ++j) {
        LinearExpr task_constraint;
        for (int i = 0; i < num_agents; ++i) {
            task_constraint += x[i][j];
        }
        solver->MakeRowConstraint(task_constraint == 1, "task_" + std::to_string(j) + "_exactly_one_agent");
    }

    // Mandatory assignment for specific agents (12, 50, 73)
    for (int i : {12, 50, 73}) {
        LinearExpr mandatory_constraint;
        for (int j = 0; j < num_tasks; ++j) {
            mandatory_constraint += x[i][j];
        }
        solver->MakeRowConstraint(mandatory_constraint == 1, "mandatory_agent_" + std::to_string(i));
    }

    // Solve the problem
    const MPSolver::ResultStatus result_status = solver->Solve();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (result_status != MPSolver::OPTIMAL) {
        std::cerr << "The problem does not have an optimal solution!" << std::endl;
        return 1;
    }

    // Output the results
    std::cout << "Optimal Total Cost: " << objective->Value() << std::endl;
    std::cout << "Optimal Assignments:" << std::endl;
    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_tasks; ++j) {
            if (x[i][j]->solution_value() > 0.5) {
                std::cout << "Agent " << i << " assigned to Task " << j
                          << " with Cost " << cost_matrix[i][j] << std::endl;
            }
        }
    }

    // Print elapsed time
    std::cout << "Time to solve: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
