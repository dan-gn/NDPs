'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

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
    
