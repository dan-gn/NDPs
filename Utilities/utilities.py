'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import json
import os
import numpy as np
import pandas as pd
import torch.nn as nn


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
NDP Model utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def get_number_of_model_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def cosine_similarity(nodes_states:np.array) -> np.array:
    # Normalise each state 
    state_norms = np.linalg.norm(nodes_states, axis=1, keepdims=True)
    state_norms[state_norms == 0] = 1.0
    normalised_states = nodes_states / state_norms

    # Compute similarity (cosine similarity)
    similarity = normalised_states @ normalised_states.T
    return similarity


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Experiment utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
def append_line_to_csv(file_path, new_line_dict):
    if os.path.exists(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
        # Append the new line (as a dictionary)
        df = pd.concat([df, pd.DataFrame([new_line_dict])])
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame([new_line_dict])
    # Save it to CSV
    df.to_csv(file_path, index=False)


def graph_reachability_fields(graph, task):
    if graph is None:
        return {
            "best_graph_are_all_outputs_reachable": None,
            "best_graph_n_unreachable_outputs": None,
            "best_graph_unreachable_output_ids": None,
        }

    unreachable_output_ids = graph.get_unreachable_outputs(
        task.parameters["graph_n_inputs"],
        task.parameters["graph_n_outputs"],
    )
    unreachable_output_ids = [int(node_id) for node_id in unreachable_output_ids]
    return {
        "best_graph_are_all_outputs_reachable": len(unreachable_output_ids) == 0,
        "best_graph_n_unreachable_outputs": len(unreachable_output_ids),
        # JSON keeps the list unambiguous after the record is written to CSV.
        "best_graph_unreachable_output_ids": json.dumps(unreachable_output_ids),
    }


def create_experiment_log(
    output_filename,
    optimisation_algorithm,
    task,
    seed,
    optimiser,
    elapsed_time,
):
    new_line = {
        "filename": output_filename,
        "algorithm": optimisation_algorithm,
        "task": task.name,
        "initial_node_state_mode": task.parameters["initial_node_state_mode"],
        "seed": seed,
        "model": task.parameters["model"],
        "hebbian": task.parameters["hebbian"],
        "time": elapsed_time,
        "configured_generations": task.parameters["generations"],
        "configured_optimisation_evaluations": task.parameters["population_size"] * (task.parameters["generations"] + 1),
        "optimisation_evaluations": optimiser.optimisation_evaluations,
        "training_reevaluations": optimiser.training_reevaluations,
        "heldout_evaluations": optimiser.heldout_evaluations,
        "total_objective_calls": optimiser.optimisation_evaluations + optimiser.training_reevaluations + optimiser.heldout_evaluations,
    }

    if optimisation_algorithm == "CMA":
        training_result = optimiser.best_training_result
        testing_result = optimiser.best_test_result

        training_graph = training_result[2] if isinstance(training_result, tuple) else None

        new_line.update(
            {
                "population_size": optimiser.es.popsize,
                "optimiser_iterations": optimiser.es.countiter,
                "best_score_mean": optimiser.best_loss,
                "best_score_test": optimiser.best_loss_test,
                "best_graph": training_result[3] if isinstance(training_result, tuple) else None,
                "best_graph_test": testing_result[3] if isinstance(testing_result, tuple) else None,
                "best_graph_n_nodes": training_graph.number_of_nodes() if training_graph is not None else None,
                "best_graph_used_nodes": training_graph.get_number_of_used_nodes() if training_graph is not None else None,
                "best_graph_n_edges": training_graph.number_of_edges() if training_graph is not None else None,
                "best_graph_used_edges": training_graph.get_number_of_used_edges() if training_graph is not None else None,
                **graph_reachability_fields(training_graph, task),
                "n_variables": len(optimiser.best_params),
                "max_stagnment": None,
                "stop_on_target": False,
                "goal_achieved": None,
                "stop_reason": ", ".join(optimiser.es.stop().keys()),
            }
        )

    elif optimisation_algorithm == "EA":
        best_individual = optimiser.best_individual
        training_graph = best_individual.best_graph

        new_line.update(
            {
                "population_size": optimiser.population_size,
                "optimiser_iterations": optimiser.i + 1,
                "best_score_mean": best_individual.fitness,
                "best_score_test": best_individual.fitness_test,
                "best_graph": best_individual.best_graph_fitness,
                "best_graph_test": best_individual.best_graph_fitness_test,
                "best_graph_n_nodes": training_graph.number_of_nodes() if training_graph is not None else None,
                "best_graph_n_edges": training_graph.number_of_edges() if training_graph is not None else None,
                **graph_reachability_fields(training_graph, task),
                "n_variables": optimiser.n_variables,
                "max_stagnment": optimiser.max_stagnment,
                "stop_on_target": optimiser.stop_on_target,
                "goal_achieved": optimiser.goal_achieved,
                "stop_reason": "target_achieved" if optimiser.goal_achieved and optimiser.stop_on_target else "completed_budget",
            }
        )

    else:
        raise ValueError("optimisation_algorithm must be 'CMA' or 'EA'.")

    return new_line


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Google colab utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
    
