import os
import yaml

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



def majority_vote(neighbors: list[int]) -> int:
    # Return the majority class label from a list of neighbors
    frequency = {}
    
    # Count the frequency of each label
    for label in neighbors:
        frequency[label] = frequency.get(label, 0) + 1
    
    # Handle the tie scenario explicitly (first encountered label wins)
    return max(frequency, key=lambda k: frequency[k])



def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs


def read_file(file_path: str) -> tuple[list[list[float]], list[int]]:
    #Read the Ionosphere dataset file and return features and labels

    #Args: file_path (str): Path to the dataset file

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
            
            # Extract label (last column: 'g' -> 1, 'b' -> 0)
            label = 1 if components[-1] == "g" else 0
            labels.append(label)
    
    return features, labels
