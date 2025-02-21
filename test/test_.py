import pytest
from pyclassify.utils import distance, distance_numpy, majority_vote
from pyclassify.numba_compiled import distance_numba
from pyclassify.classifier import kNN
import numpy as np

# Test distance function
def test_distance():
    # Test distance between identical points
    assert distance([0, 0], [0, 0]) == 0.0
    # Test distance between two points in 2D space
    assert distance([1, 2], [4, 6]) == 5.0
    # Test distance between points with negative values
    assert distance([-1, -2], [-4, -6]) == 5.0
    # Test distance between points with mismatched dimensions
    with pytest.raises(ValueError):
        distance([1, 2], [1, 2, 3])

    # Symmetry Test
    assert distance([1, 2], [4, 6]) == distance([4, 6], [1, 2])
    
    # Triangular Inequality Test
    assert distance([1, 2], [4, 6]) <= distance([1, 2], [2, 3]) + distance([2, 3], [4, 6])

def test_majority_vote():
    # Test case with a clear majority
    assert majority_vote([1, 0, 0, 0]) == 0
    # Test case with a tie (first encountered label is returned)
    assert majority_vote([1, 1, 0, 0]) in [0, 1]
    # Test case with a single label
    assert majority_vote([1]) == 1
    # Test case with all labels the same
    assert majority_vote([2, 2, 2, 2]) == 2
    # Test case with a tie (even split, should return the first encountered label)
    assert majority_vote([0, 0, 1, 1]) in [0, 1]

def test_kNN_constructor():
    # Test valid k value
    classifier = kNN(k=3)
    assert classifier.k == 3
    # Test invalid k value (non-integer)
    with pytest.raises(TypeError):
        kNN(k="3")
    with pytest.raises(TypeError):
        kNN(k=3.5)
    # Test invalid k value (negative, zero)
    with pytest.raises(ValueError):
        kNN(k=0)
    with pytest.raises(ValueError):
        kNN(k=-1)


@pytest.mark.parametrize("backend", ["plain", "numpy", "numba"])
def test_knn_backend(backend):
    # Test with specified backend
    knn = kNN(k=2, backend=backend)
    X = [[1, 2], [2, 3], [8, 9]]
    y = [0, 0, 1]
    new_points = [[1, 2], [2, 3]]
    predictions = knn((X, y), new_points)
    assert predictions == [0, 0], f"Predictions should match expected classes for backend={backend}"

def test_knn_distance_functions():
    # Test distance functions
    point1 = [1, 2]
    point2 = [3, 4]
    assert distance(point1, point2) == distance_numpy(np.array(point1), np.array(point2)), "Distance functions should match"

def test_knn_edge_cases():
    knn = kNN(k=2, backend="numpy")
    X = [[1, 2], [2, 3], [8, 9]]
    y = [0, 0, 1]

    # Test a point equidistant to both classes
    predictions = knn((X, y), [[5, 6]])
    assert predictions[0] in [0, 1], "Equidistant point should return a valid class"

    # Test a point outside the training data range
    predictions = knn((X, y), [[10, 11]])
    assert predictions == [1], "Point outside training data range should return the nearest class"

def test_distance_numba():
    # Test with small arrays
    point1 = np.array([1.0, 2.0, 3.0])
    point2 = np.array([4.0, 5.0, 6.0])
    assert np.isclose(distance_numba(point1, point2), 5.196152422706632)

    # Test with large arrays
    point1 = np.random.rand(1000)
    point2 = np.random.rand(1000)
    expected = np.linalg.norm(point1 - point2)
    assert np.isclose(distance_numba(point1, point2), expected)


def main():
    # Run all the tests
    test_distance()
    test_majority_vote()
    test_kNN_constructor()
    for backend in ["plain", "numpy"]:
        test_knn_backend(backend)
    test_knn_distance_functions()
    test_knn_edge_cases()
    test_distance_numba()
    print("All tests passed successfully!")

# Call the main function to run tests
if __name__ == "__main__":
    main()