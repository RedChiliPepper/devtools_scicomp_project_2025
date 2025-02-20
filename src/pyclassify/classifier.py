from .utils import majority_vote, distance

class kNN:
    def __init__(self, k: int):

        # Check if k is integer
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        # Check if k is positive
        if k <= 0:
            raise ValueError("k must be a positive integer")   

        # Initialize the kNN classifier with the number of neighbors k
        self.k = k 

    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x: list[float]) -> list[int]:
        # Return the labels of the k nearest neighbors of x in the dataset (X, y)
        # Calculate Euclidean distance between x and each point in X
        distances = []
        for i, point in enumerate(X):
            dist = distance(point, x)
            distances.append((dist, y[i]))
        
        # Sort by distance and select the k nearest neighbors
        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        return k_nearest_labels

    def __call__(self, data: tuple[list[list[float]], list[int]], new_points: list[list[float]]) -> list[int]:
        # Classify each point in new_points using the kNN algorithm
        X, y = data
        predictions = []
        
        for point in new_points:
            # Get the k nearest neighbors
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            # Perform majority voting
            predicted_class = majority_vote(neighbors)
            predictions.append(predicted_class)
        return predictions

