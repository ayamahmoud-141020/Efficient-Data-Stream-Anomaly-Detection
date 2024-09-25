# main.py

from model_prod import model
from anomaly import anomaly_dect

if __name__ == '__main__':
    model()  # Train the model and save it
    anomaly_dect()  # Start anomaly detection
