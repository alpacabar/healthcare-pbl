import tensorflow as tf
import tensorflow_federated as tff

# Define a simple model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),  # Input features: age, gender, ICD code
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: Glucose level prediction
    ])
    return model

# Wrap the model for TFF
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec={'x': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                    'y': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)},
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
