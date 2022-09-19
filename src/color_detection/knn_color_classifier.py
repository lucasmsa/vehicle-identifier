import os
import csv
import math


class KNNColorClassifier:
    K = 3

    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        self.training_feature_vector = []
        self.test_feature_vector = []

    def run(self):
        self.extract_feature_vectors_from_data_files()
        self.fetch_classifier_predictions()

    def calculate_euclidean_distance(self, first_variable, second_variable, length):
        distance = 0
        for idx in range(length):
            distance += pow(first_variable[idx] - second_variable[idx], 2)
        return math.sqrt(distance)

    def execute_knn_algorithm(self, test_instance):
        euclidean_distances, test_instance_size, nearest_neighbors = [], len(
            test_instance), []

        for training_instance_idx in range(len(self.training_feature_vector)):
            euclidean_distance = self.calculate_euclidean_distance(
                test_instance, self.training_feature_vector[training_instance_idx], test_instance_size)
            euclidean_distances.append(
                (self.training_feature_vector[training_instance_idx], euclidean_distance))
        euclidean_distances.sort(key=lambda x: x[1])
        for neighbor_idx in range(self.K):
            nearest_neighbors.append(euclidean_distances[neighbor_idx][0])
        return nearest_neighbors

    def fetch_nearest_neighbors_votes(self, nearest_neighbors):
        neighbors_responses = {}

        for neighbor_idx in range(len(nearest_neighbors)):
            neighbor_response = nearest_neighbors[neighbor_idx][-1]
            if neighbor_response in neighbors_responses:
                neighbors_responses[neighbor_response] += 1
            else:
                neighbors_responses[neighbor_response] = 1

        sorted_votes = sorted(neighbors_responses.items(),
                              key=lambda x: x[1], reverse=True)

        return sorted_votes[0][0]

    def fetch_classifier_predictions(self):
        classifier_predictions = []
        for test_instance_idx in range(len(self.test_feature_vector)):
            nearest_neighbors = self.execute_knn_algorithm(
                self.test_feature_vector[test_instance_idx])
            result = self.fetch_nearest_neighbors_votes(nearest_neighbors)
            classifier_predictions.append(result)

        return classifier_predictions[0]

    def extract_feature_vectors_from_data_files(self):
        with open(self.training_data) as training_file:
            self.parse_rgb_values(self.training_feature_vector, training_file)

        with open(self.test_data) as testing_file:
            self.parse_rgb_values(self.test_feature_vector, testing_file)

    def parse_rgb_values(self, feature_vector, file):
        lines = csv.reader(file)
        dataset = list(lines)

        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            feature_vector.append(dataset[x])


current_directory_path = f"{os.path.dirname(os.path.realpath(__file__))}"
knn_color_classifier = KNNColorClassifier(
    f"{current_directory_path}/training.data", f"{current_directory_path}/test.data")
knn_color_classifier.run()
