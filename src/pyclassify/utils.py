import os
import yaml
import numpy as np
from line_profiler import profile
from numba import njit, types
from numba.pycc import CC

@profile
def distance(point1: list[float], point2: list[float]) -> float:
    # Computing the Euclidean distance between two points
    # Arguments: two points as lists, Returns: squared Euclidean distance

    # Initialize the sum of squared differences
    sum_of_squares = 0.0
    
    # Raise error if dimensions not matching
    if len(point1) != len(point2):
        raise ValueError("The dimensions of the points do not match")
    
    # Loop through the points and calculate the sum of squared differences
    for p1, p2 in zip(point1, point2):
        sum_of_squares += (p1 - p2) ** 2
    
    # Return the Euclidean distance
    return sum_of_squares ** 0.5

@profile
def distance_numpy(point1: np.ndarray, point2: np.ndarray) -> float:
    # Computing the Euclidean distance between two points using NumPy
    # Arguments: two points as NumPy arrays, Returns: Euclidean distance
    return np.sqrt(np.linalg.norm(point1 - point2))


@profile
def majority_vote(neighbors: list[int]) -> int:
    # Return the majority class label from a list of neighbors
    frequency = {}
    
    # Count the frequency of each label
    for label in neighbors:
        frequency[label] = frequency.get(label, 0) + 1
    
    # Handle the tie scenario explicitly (first encountered label wins)
    return max(frequency, key=lambda k: frequency[k])

# Initialize the compiler
cc = CC('numba_compiled')

# Define the optimized distance function
@profile
@cc.export('distance_numba', 'f8(f8[:], f8[:])')
def distance_numba(point1: np.ndarray, point2: np.ndarray) -> float:
    # Compute the Euclidean distance manually
    diff = point1 - point2
    return np.sqrt(np.sum(diff ** 2))

# Compile the module
#if __name__ == "__main__":
cc.compile()


def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs


def read_file(file_path: str) -> tuple[list[list[float]], list[int]]:
    # Read a dataset file and return features and labels
    # Supports both Ionosphere (labels 'g'/'b') and Spambase (binary labels 0/1)

    # Args: file_path (str): Path to the dataset file

    # Returns: tuple[list[list[float]], list[int]]: A tuple containing:
    # features: A list of feature vectors (each vector is a list of floats)
    # labels: A list of integer labels (0 or 1)

    features = []
    labels = []

    with open(file_path, "r") as file:
        for line in file:
            # Split the line into components
            components = line.strip().split(",")
            
            # Extract features (all columns except the last)
            feature_vector = [float(x) for x in components[:-1]]
            features.append(feature_vector)
            
            # Extract label
            label_str = components[-1].strip()
            if label_str in ["g", "b"]:  # Ionosphere dataset
                label = 1 if label_str == "g" else 0
            else:  # Spambase dataset (binary labels 0/1)
                label = int(float(label_str))
            labels.append(label)
    
    return features, labels