import pytest
from pyclassify.utils import distance, majority_vote
from pyclassify.classifier import kNN

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

def main():
    # Run all the tests
    test_distance()
    test_majority_vote()
    test_kNN_constructor()
    print("All tests passed successfully!")

# Call the main function to run tests
if __name__ == "__main__":
    main()