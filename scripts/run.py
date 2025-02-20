import argparse

from pyclassify.classifier import kNN
from pyclassify.utils import read_file, read_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the configuration file (optional)", default=None)
    args = parser.parse_args()
    config_path = args.config

    # Step 1: Read the configuration file or use defaults if no config path is provided
    if config_path:
        config = read_config(config_path)
    else:
        # Default config if no config path is provided
        config = {
            "dataset": "./data/ionosphere.data",  # Default dataset path
            "k": 5  # Default k value for kNN
        }
    dataset_path = config.get("dataset", "./data/ionosphere.data") # Read dataset path from config
    k = config.get("k", 5)

    # Set default values for test_size and train_size
    test_size = config.get("test_size", 0.2)  # Default test_size is 0.2 (20%)
    train_size = config.get("train_size", 0.8)  # Default train_size is 0.8 (80%)

    # Validate test_size and train_size
    if test_size + train_size != 1.0:
        raise ValueError("test_size and train_size must sum to 1.0")

    # Step 2: Read the dataset
    features, labels = read_file(dataset_path)

    # Step 3: Manually split the dataset into train and test sets
    split_index = int(len(features) * (1 - test_size))
    X_train = features[:split_index]
    X_test = features[split_index:]
    y_train = labels[:split_index]
    y_test = labels[split_index:]

    # Step 4: Perform kNN classification
    classifier = kNN(k=k)
    predictions = classifier((X_train, y_train), X_test)  # Use __call__ method

    # Step 5: Compute and print accuracy
    # Convert y_test and predictions to integers (if they are not already)
    y_test = [int(label) for label in y_test]
    predictions = [int(prediction) for prediction in predictions]

    accuracy = sum(pred == true for pred, true in zip(predictions, y_test)) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

