# Utilities

This README provides an overview of the Python script that includes functions for various data preprocessing and model artifact saving tasks. 
The script utilizes the pandas, joblib, logging, and json libraries, and it is designed to facilitate common data science tasks.

- `read_datasets`: Reads datasets from CSV files and returns pandas DataFrames for the training, validation, and test datasets.
- `split_xy`: Splits a DataFrame into features (X) and labels (y).
- `indices_in_list`: Retrieves the indices of specific elements in a base list
- `save_model_artifact`: Saves a trained model to a file.
- `save_metrics`: Saves evaluation metrics to a JSON file.
- `save_training_dataset_metadata`: Saves training dataset information for model monitoring.
