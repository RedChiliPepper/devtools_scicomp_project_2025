import numpy as np
import sys
import os
from .utils import majority_vote, distance, distance_numpy
from line_profiler import profile

#from numba_compiled import distance_numba  # Import the compiled function
from .numba_compiled import distance_numba

# Add this at the top of the file
LINE_PROFILE = 1

if LINE_PROFILE:
    profile = profile
else:
    profile = lambda x: x

class kNN:
    def __init__(self, k: int, backend: str = "plain"):
        # Check if k is integer
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        # Check if k is positive
        if k <= 0:
            raise ValueError("k must be a positive integer")
        
        # Check if backend is valid
        if backend not in ["plain", "numpy", "numba"]:
            raise ValueError("backend must be 'plain', 'numpy', or 'numba'")   

        # Initialize the kNN classifier with the number of neighbors k
        self.k = k 
        self.backend = backend
        self.distance = distance if backend == "plain" else (distance_numpy if backend == "numpy" else distance_numba)

    @profile
    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x: list[float]) -> list[int]:
        # Return the labels of the k nearest neighbors of x in the dataset (X, y)
        # Calculate Euclidean distance between x and each point in X
        distances = []
        for i, point in enumerate(X):
            dist = self.distance(point, x)
            distances.append((dist, y[i]))
        
        # Sort by distance and select the k nearest neighbors
        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        return k_nearest_labels
    
    @profile
    def __call__(self, data: tuple[list[list[float]], list[int]], new_points: list[list[float]]) -> list[int]:
        # Classify each point in new_points using the kNN algorithm
        X, y = data
        predictions = []

        # Convert X to NumPy arrays if backend is "numpy" or "numba"
        if self.backend in ["numpy", "numba"]:
            X = [np.array(x) for x in X]

        for point in new_points:
            # Convert point to NumPy array if backend is "numpy" or "numba"
            if self.backend in ["numpy", "numba"]:
                point = np.array(point)
        
            # Get the k nearest neighbors
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            # Perform majority voting
            predicted_class = majority_vote(neighbors)
            predictions.append(predicted_class)
        return predictions