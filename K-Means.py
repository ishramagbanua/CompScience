class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.labels = []

    def fit(self, X):
        """Fit the K-Means model to the data."""
        # Randomly initialize centroids
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            # Assign clusters
            self.labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._update_centroids(X)

            # Check for convergence
            if self._has_converged(new_centroids):
                break

            self.centroids = new_centroids

    def _initialize_centroids(self, X):
        """Randomly initialize centroids from the dataset."""
        import random
        return [X[random.randint(0, len(X) - 1)] for _ in range(self.k)]

    def _assign_clusters(self, X):
        """Assign each data point to the nearest centroid."""
        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels

    def _update_centroids(self, X):
        """Update centroid positions based on current cluster assignments."""
        new_centroids = []
        for i in range(self.k):
            cluster_points = [X[j] for j in range(len(X)) if self.labels[j] == i]
            if cluster_points:
                new_centroids.append(self._mean(cluster_points))
            else:
                new_centroids.append(self.centroids[i])  # Keep the old centroid if no points
        return new_centroids

    def _mean(self, points):
        """Calculate the mean of a list of points."""
        return [sum(dim) / len(points) for dim in zip(*points)]

    def _euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5

    def _has_converged(self, new_centroids):
        """Check if centroids have changed."""
        for old, new in zip(self.centroids, new_centroids):
            if old != new:
                return False
        return True

    def predict(self, X):
        """Predict the closest cluster for each data point."""
        return self._assign_clusters(X)


# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]

    # Create KMeans classifier
    kmeans = KMeans(k=2)

    # Fit the model
    kmeans.fit(X)

    # Predict the clusters
    labels = kmeans.predict(X)

    print("Cluster centers:", kmeans.centroids)
    print("Predicted labels:", labels)
