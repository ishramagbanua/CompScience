import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """Store the training dataset."""
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Predict the class labels for the provided data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        """Predict the class label for a single example."""
        # Compute distances between x and all examples in the training set
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Create KNN classifier
    knn = KNN(k=3)
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    # Sample test data
    X_test = np.array([[1, 2], [5, 5], [8, 7]])
    
    # Predict the classes
    predictions = knn.predict(X_test)
    
    print("Predicted classes:", predictions)
