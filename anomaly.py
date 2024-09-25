"""
This module implements real-time anomaly detection on a continuous data stream.

It uses the Isolation Forest algorithm for anomaly detection and includes
functionality for adaptive learning and real-time visualization of the data stream.

Key components:
- Data stream simulation
- Anomaly detection using Isolation Forest
- Adaptive learning to handle concept drift
- Real-time visualization of normal and anomalous data points
"""
import os
import random
from datetime import datetime
from joblib import load, dump
import logging
import matplotlib.pyplot as plt
import numpy as np
from settings import DELAY, OUTLIERS_GENERATION_PROBABILITY, VISUALIZATION

logging.basicConfig(filename='anomaly.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

data_ls = []
anomaly_count = 0
total_count = 0

def set_dark_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#1e1e1e'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

def validate_data(data):
    """
    التحقق من صحة البيانات الواردة
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError("Data must be a 2D array with shape (n, 1)")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain numeric values")

def adaptive_learning(clf, scaler, new_data, update_frequency=1000):
    """
    تحديث النموذج باستخدام البيانات الجديدة
    """
    X_new = np.array(new_data)
    X_new_scaled = scaler.transform(X_new)
    
    # إعادة تدريب النموذج بالكامل
    clf.fit(X_new_scaled)
    
    dump((clf, scaler), './isolation_forest_model.joblib')
    print(f"Model updated with {len(new_data)} new data points")
    return clf, scaler

def anomaly_dect():
    global anomaly_count, total_count
    _id = 0
    new_data = []

    try:
        clf, scaler = load('./isolation_forest_model.joblib')
    except FileNotFoundError:
        logging.error("Model file not found. Please train the model first.")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    if VISUALIZATION:
        set_dark_style()
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(12, 7))
        normal_plot, = ax.plot([], [], 'c.', label='Normal', alpha=0.6, markersize=8)
        anomaly_plot, = ax.plot([], [], 'r.', label='Anomaly', alpha=0.8, markersize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-10, 10)
        ax.set_title("Real-time Anomaly Detection", fontsize=16, fontweight='bold')
        ax.set_xlabel("Data points", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        background = ax.fill_between(ax.get_xlim(), -10, 10, color='#2a2a2a', alpha=0.3)
        plt.tight_layout()

    while True:
        try:
            if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
                X_test = np.random.uniform(low=-10, high=10, size=(1, 1))
            else:
                X_test = np.random.normal(loc=0, scale=1, size=(1, 1))

            X_test = np.round(X_test, 3)
            validate_data(X_test)

            current_time = datetime.utcnow().isoformat()

            record = {"id": _id, "data": X_test.tolist(), "current_time": current_time}
            print(f"Incoming: {record}")

            scaled_data = scaler.transform(X_test)
            prediction = clf.predict(scaled_data)
            score = clf.score_samples(scaled_data)

            total_count += 1
            if prediction[0] == -1:
                anomaly_count += 1
                print(f"\033[91mANOMALY DETECTED: {record}\033[0m")  # Red color in terminal
                if VISUALIZATION:
                    anomaly_plot.set_data(np.append(anomaly_plot.get_xdata(), _id), 
                                          np.append(anomaly_plot.get_ydata(), X_test[0][0]))
            else:
                if VISUALIZATION:
                    normal_plot.set_data(np.append(normal_plot.get_xdata(), _id), 
                                         np.append(normal_plot.get_ydata(), X_test[0][0]))

            print(f"Incoming data: {X_test.tolist()}, Prediction: {prediction[0]}, Anomaly Score: {score[0]:.4f}")
            logging.info(f"Incoming data: {X_test.tolist()}, Prediction: {prediction[0]}, Anomaly Score: {score[0]:.4f}")

            if _id % 10 == 0:
                anomaly_rate = (anomaly_count / total_count) * 100 if total_count > 0 else 0
                print(f"\n\033[1mSummary after {total_count} data points:\033[0m")
                print(f"Anomalies detected: {anomaly_count}")
                print(f"Anomaly rate: {anomaly_rate:.2f}%\n")

            if VISUALIZATION:
                ax.set_xlim(_id - 100 if _id > 100 else 0, _id + 1)
                background.remove()
                background = ax.fill_between(ax.get_xlim(), -10, 10, color='#2a2a2a', alpha=0.3)
                fig.canvas.draw()
                fig.canvas.flush_events()

            new_data.append(X_test)
            if len(new_data) >= 1000:
                clf, scaler = adaptive_learning(clf, scaler, new_data)
                new_data = []  # Reset new_data after updating the model

            _id += 1
            plt.pause(DELAY)

        except ValueError as ve:
            logging.error(f"Data validation error: {ve}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            continue

    if VISUALIZATION:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    anomaly_dect()