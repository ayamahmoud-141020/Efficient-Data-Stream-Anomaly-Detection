"""
This module contains unit tests for the anomaly detection system.

It tests various components of the system including model creation,
data validation, anomaly detection, and adaptive learning.
"""
import unittest
import numpy as np
from model_prod import model
from anomaly import validate_data, adaptive_learning
from joblib import load
import os

class TestAnomalyDetection(unittest.TestCase):
    def setUp(self):
        model()
        self.clf, self.scaler = load('./isolation_forest_model.joblib')

    def test_model_creation(self):
        self.assertTrue(os.path.exists('./isolation_forest_model.joblib'))
        self.assertIsNotNone(self.clf)
        self.assertIsNotNone(self.scaler)

    def test_normal_data_detection(self):
        normal_data = np.array([[0.1], [1.0], [2.0], [-1.0], [-2.0]])
        scaled_data = self.scaler.transform(normal_data)
        predictions = self.clf.predict(scaled_data)
        self.assertTrue(all(pred == 1 for pred in predictions))

    def test_anomaly_data_detection(self):
        anomaly_data = np.array([[10.0], [-10.0], [20.0], [-20.0]])
        scaled_data = self.scaler.transform(anomaly_data)
        predictions = self.clf.predict(scaled_data)
        self.assertTrue(all(pred == -1 for pred in predictions))

    def test_outliers_generation_probability(self):
        from settings import OUTLIERS_GENERATION_PROBABILITY
        self.assertTrue(0 <= OUTLIERS_GENERATION_PROBABILITY <= 1)

    def test_validate_data(self):
        # Test valid data
        valid_data = np.array([[1.0], [2.0]])
        self.assertIsNone(validate_data(valid_data))

        # Test invalid data types
        with self.assertRaises(ValueError):
            validate_data([1, 2, 3])
        with self.assertRaises(ValueError):
            validate_data(np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            validate_data(np.array([[1, 2], [3, 4]]))

    def test_adaptive_learning(self):
        # Get initial predictions for a range of values
        test_range = np.linspace(-10, 10, 100).reshape(-1, 1)
        initial_predictions = self.clf.predict(self.scaler.transform(test_range))

        # Print some initial information
        print("Initial predictions (first 10):", initial_predictions[:10])
        print("Initial decision function (first 5):", self.clf.decision_function(self.scaler.transform(test_range[:5])))

        # Generate new data that's significantly different
        new_data = np.random.uniform(low=-20, high=20, size=(1000, 1))

        # Apply adaptive learning
        self.clf, self.scaler = adaptive_learning(self.clf, self.scaler, new_data, update_frequency=1)

        # Get updated predictions
        updated_predictions = self.clf.predict(self.scaler.transform(test_range))

        # Print updated information
        print("Updated predictions (first 10):", updated_predictions[:10])
        print("Updated decision function (first 5):", self.clf.decision_function(self.scaler.transform(test_range[:5])))

        # Check if there's any difference in predictions
        prediction_changed = not np.array_equal(initial_predictions, updated_predictions)
        self.assertTrue(prediction_changed, "Adaptive learning did not change the model's predictions")

        # Additional check: ensure some predictions changed
        diff_count = np.sum(initial_predictions != updated_predictions)
        print(f"Number of changed predictions: {diff_count}")
        self.assertGreater(diff_count, 0,
                           f"No predictions changed after adaptive learning. Differences: {diff_count}")

    def test_score_samples(self):
        normal_data = np.array([[0.1]])
        anomaly_data = np.array([[100.0]])
        normal_score = self.clf.score_samples(self.scaler.transform(normal_data))
        anomaly_score = self.clf.score_samples(self.scaler.transform(anomaly_data))
        self.assertGreater(normal_score, anomaly_score)

if __name__ == '__main__':
    unittest.main()