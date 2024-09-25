"""
This module is responsible for creating and training the Isolation Forest model
for anomaly detection. It also saves the trained model for later use.

The Isolation Forest algorithm is particularly effective for anomaly detection
as it can handle high-dimensional datasets and does not require a labeled dataset.
"""
import numpy as np
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def model():
    """
    Trains an Isolation Forest model and saves it to a file.
    """
    rng = np.random.RandomState(100)

    # Generating random training data
    normal_data = rng.normal(loc=0, scale=1, size=(5000, 1))
    outliers = rng.uniform(low=-10, high=10, size=(250, 1))
    X_train = np.vstack((normal_data, outliers))

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit the Isolation Forest model with refined parameters
    clf = IsolationForest(n_estimators=200,
                          max_samples='auto',
                          contamination=0.05,
                          max_features=1,
                          bootstrap=True,
                          n_jobs=-1,
                          random_state=rng)
    clf.fit(X_train_scaled)

    # Save the model and scaler
    dump((clf, scaler), './isolation_forest_model.joblib')
    logging.info("Model and scaler trained and saved to 'isolation_forest_model.joblib'.")

if __name__ == "__main__":
    model()