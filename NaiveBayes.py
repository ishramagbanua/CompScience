import math

class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.likelihood = {}
        self.vocabulary = set()
        self.total_docs = 0

    def fit(self, X, y):
        self.total_docs = len(X)
        self.prior = {}
        self.likelihood = {}

        # Count occurrences
        class_counts = {}
        word_counts = {}

        for i in range(len(X)):
            label = y[i]
            if label not in class_counts:
                class_counts[label] = 0
                word_counts[label] = {}
            class_counts[label] += 1

            for word in X[i].split():
                self.vocabulary.add(word)
                if word not in word_counts[label]:
                    word_counts[label][word] = 0
                word_counts[label][word] += 1

        # Calculate prior and likelihood
        for label, count in class_counts.items():
            self.prior[label] = count / self.total_docs
            self.likelihood[label] = {}
            total_words = sum(word_counts[label].values())

            for word in self.vocabulary:
                word_freq = word_counts[label].get(word, 0)
                self.likelihood[label][word] = (word_freq + 1) / (total_words + len(self.vocabulary))  # Laplace smoothing

    def predict(self, X):
        predictions = []

        for document in X:
            class_probabilities = {}
            for label in self.prior:
                class_probabilities[label] = math.log(self.prior[label])  # Use log for numerical stability

                for word in document.split():
                    if word in self.likelihood[label]:
                        class_probabilities[label] += math.log(self.likelihood[label][word])

            # Choose the class with the highest probability
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)

        return predictions

# Example usage
if _name_ == "_main_":
    # Sample dataset
    X = [
        "cheap meds",
        "buy now",
        "hello friend",
        "meeting tomorrow",
        "free trial",
        "let's catch up",
        "limited time offer",
        "see you soon"
    ]
    y = [
        "spam",
        "spam",
        "not spam",
        "not spam",
        "spam",
        "not spam",
        "spam",
        "not spam"
    ]

    # Initialize and train the model
    model = NaiveBayes()
    model.fit(X, y)

    # Predict new instances
    test_data = [
        "cheap offer",
        "let's meet up",
        "buy meds now"
    ]
    predictions = model.predict(test_data)

    # Output predictions
    for doc, pred in zip(test_data, predictions):
        print(f"Document: '{doc}' => Predicted: '{pred}'")
