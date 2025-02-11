class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """Store the training dataset."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class labels for the provided data."""
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return predictions

    def _predict(self, x):
        """Predict the class label for a single example."""
        distances = []
        for i in range(len(self.X_train)):
            distance = self._euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        # Sort by distance and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_labels = [distances[i][1] for i in range(self.k)]

        # Return the most common class label
        return self._most_common(nearest_labels)

    def _euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5

    def _most_common(self, labels):
        """Return the most common label among the neighbors."""
        count = {}
        for label in labels:
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
        return max(count, key=count.get)


# Example usage
if __name__ == "__main__":
    # Sample dataset
    X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 6]]
    y_train = [0, 0, 0, 1, 1, 1]

    # Create KNN classifier
    knn = KNN(k=3)

    # Fit the model
    knn.fit(X_train, y_train)

    # Sample test data
    X_test = [[1, 2], [5, 5], [8, 7]]

    # Predict the classes
    predictions = knn.predict(X_test)

    print("Predicted classes:", predictions)
