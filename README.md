# Efficient Data Stream Anomaly Detection

## Overview

This project implements a real-time anomaly detection system for continuous data streams using Python. It utilizes the Isolation Forest algorithm to identify unusual patterns or exceptionally high values in streaming data, which could represent various metrics such as financial transactions or system performance indicators.

## Key Features

- **Real-time Anomaly Detection**: Continuously analyzes incoming data points to flag anomalies.
- **Adaptive Learning**: Implements a mechanism to adapt to concept drift and seasonal variations in the data.
- **Data Stream Simulation**: Includes a function to emulate a real-time data stream with configurable parameters.
- **Visualization**: Provides a real-time visualization tool to display both the data stream and detected anomalies.
- **Optimized Performance**: Designed for both speed and efficiency in processing continuous data streams.
- **Robust Error Handling**: Includes comprehensive error handling and data validation.

## Technology Stack

- Python 3.x
- NumPy for numerical operations
- scikit-learn for the Isolation Forest algorithm
- Matplotlib for real-time visualization
- Joblib for model persistence

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/efficient-data-stream-anomaly-detection.git
   cd efficient-data-stream-anomaly-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the initial model:
   ```
   python model_prod.py
   ```

2. Run the anomaly detection system:
   ```
   python anomaly.py
   ```

3. To run the tests:
   ```
   python -m unittest test_anomaly_detection.py
   ```

## Project Structure

- `model_prod.py`: Creates and trains the initial Isolation Forest model.
- `anomaly.py`: Contains the main anomaly detection logic and real-time visualization.
- `test_anomaly_detection.py`: Includes unit tests for various components of the system.
- `settings.py`: Stores configuration parameters for the system.

## Customization

You can adjust various parameters in `settings.py` to customize the behavior of the anomaly detection system:

- `DELAY`: Controls the speed of data generation and visualization updates.
- `OUTLIERS_GENERATION_PROBABILITY`: Adjusts the frequency of generated anomalies.
- `VISUALIZATION`: Toggles the real-time visualization on or off.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

This project is open-sourced under the [MIT license](LICENSE).

## Acknowledgements

This project was developed as part of an application for the Graduate Software Engineer role at Cobblestone Energy. Special thanks to the Cobblestone Talent Acquisition Team for the opportunity to work on this challenging problem.

