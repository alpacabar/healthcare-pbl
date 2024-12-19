import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_mimic_demo
from fl_model import model_fn
from dp_utils import dp_optimizer

# Function to split data into federated clients
def create_federated_data(X, y, num_clients=3):
    client_datasets = []
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    for client_X, client_y in zip(X_splits, y_splits):
        dataset = tf.data.Dataset.from_tensor_slices((
            {'x': client_X.values.astype('float32')},
            client_y.values.reshape(-1, 1).astype('float32')
        )).batch(8)  # Batch size of 8
        client_datasets.append(dataset)
    return client_datasets

# Evaluation function
def evaluate_global_model(state, test_dataset):
    keras_model = model_fn().keras_model
    state.model.assign_weights_to(keras_model)  # Assign weights from TFF state
    results = keras_model.evaluate(test_dataset.batch(8), verbose=0)
    print(f"Evaluation results: {results}")

# Main federated learning pipeline
if __name__ == "__main__":
    # Load and preprocess data
    combined_data = load_and_preprocess_mimic_demo("data/")
    X = combined_data[['anchor_age', 'gender', 'icd_code']]
    y = combined_data['glucose_level']
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simulate federated clients
    federated_train_data = create_federated_data(X_train, y_train)

    # Prepare test data for evaluation
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'x': X_test.values.astype('float32')},
        y_test.values.reshape(-1, 1).astype('float32')
    ))

    # Define federated training process
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: dp_optimizer(learning_rate=0.01, noise_multiplier=0.5, l2_norm_clip=1.0)
    )

    # Initialize the process
    state = iterative_process.initialize()

    # Train for 10 federated rounds
    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f"Round {round_num}: {metrics}")

    # Evaluate the global model on the test dataset
    evaluate_global_model(state, test_dataset)
