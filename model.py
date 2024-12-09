import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.legacy import Nadam
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import pandas as pd

# TODO: combine/align with args from init.py?
class ExperimentConfig:
    """Configuration for fairness experiment."""
    dataset_name: str
    protected_attributes: list[str]
    target_column: str = 'Probability'
    test_size: float = 0.2
    random_seed: int = 42
    n_experiments: int = 20
    model_epochs: int = 10 

def compute_joint_probability(predictions, sensitive_labels):
    predictions = tf.reshape(predictions, [-1])
    sensitive_labels = tf.reshape(sensitive_labels, [-1])

    total_samples = tf.cast(tf.shape(predictions)[0], tf.float32)  # Batch size

    # Get unique values of predictions and sensitive labels
    unique_y, _ = tf.unique(predictions)
    unique_s, _ = tf.unique(sensitive_labels)

    # Create masks for all combinations of (y, s)
    y_masks = tf.equal(tf.expand_dims(predictions, axis=1), unique_y)  # [batch_size, num_unique_y]
    s_masks = tf.equal(tf.expand_dims(sensitive_labels, axis=1), unique_s)  # [batch_size, num_unique_s]

    # Compute joint probabilities by combining masks
    joint_counts = tf.matmul(tf.cast(y_masks, tf.float32), tf.cast(s_masks, tf.float32), transpose_a=True)
    joint_probs = joint_counts / total_samples  # Normalize to get probabilities

    joint_probs = tf.clip_by_value(joint_probs, 1e-10, 1.0)

    return joint_probs, unique_y, unique_s

def compute_marginal_probability(values):
    marginal_probs = {}
    total_samples = tf.cast(tf.shape(values)[0], tf.float32)
    # total_samples = len(values)

    values = tf.reshape(values, [-1])

    unique_values = tf.unique(values)[0]

    for value in unique_values:
        mask = tf.equal(values, value)
        marginal_count = tf.reduce_sum(tf.cast(mask, tf.float32))
        marginal_probs[float(value)] = marginal_count / total_samples
    
    return marginal_probs

def compute_prejudice_remover_loss(predictions, sensitive_labels):
    # Compute joint probabilities and unique values
    joint_probs, unique_y, unique_s = compute_joint_probability(predictions, sensitive_labels)

    # Compute marginal probabilities
    marginal_probs_y = tf.reduce_sum(joint_probs, axis=1)  # Sum over sensitive labels
    marginal_probs_s = tf.reduce_sum(joint_probs, axis=0)  # Sum over predictions

    # Compute mutual information-based loss
    joint_log_probs = joint_probs * tf.math.log(
        joint_probs / (tf.expand_dims(marginal_probs_y, axis=1) * tf.expand_dims(marginal_probs_s, axis=0) + 1e-10) + 1e-10
    )
    loss = tf.reduce_sum(joint_log_probs)  # Sum over all (y, s) pairs

    return loss

class PrejudiceRemoverLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_reg, protected_labels):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.protected_labels = tf.convert_to_tensor(protected_labels, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, 1])
        batch_sensitive_labels = self.protected_labels[:tf.shape(y_pred)[0]]

        if len(batch_sensitive_labels.shape) == 1:
            batch_sensitive_labels = tf.expand_dims(batch_sensitive_labels, axis=1)

        task_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        prejudice_loss = 0.0
        for attr_labels in tf.unstack(batch_sensitive_labels, axis=1):
            prejudice_loss += compute_prejudice_remover_loss(y_pred, attr_labels)

        return task_loss + self.lambda_reg * prejudice_loss
class Vanilla():
    """Simple Keras model"""
    
    def __init__(self, input_shape, batch_size=32, epochs=20):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Sequential:
        return keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(self.input_shape, )),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid", name="vanilla_network_output")
        ])
        
    def fit(self, X, y):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Nadam(),
            metrics=["accuracy", 
                    keras.metrics.AUC(),
                    keras.metrics.Precision(),
                    keras.metrics.Recall()]
        )
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self
    
    def predict(self, X) -> np.ndarray:
        predictions = self.model.predict(X)
        return (predictions >= 0.5).astype(int).flatten()

class Adversary(Vanilla):
    """Modifying Vanilla DNN to Adversarial Network"""

    def __init__(self, input_shape, protected_attribute_names, protected_shapes, batch_size=32, epochs=20, lambda_reg=0.1, learning_rate=2e-3, prejudice_remover=False):
        super().__init__(input_shape, batch_size, epochs)
        self.lambda_reg = lambda_reg  # lambda in combined loss formula (L = L_{vanilla} - lambda_reg * L_{adversary})
        self.protected_attribute_names = protected_attribute_names
        self.learning_rate = learning_rate
        self.protected_shapes = protected_shapes # list of number of protected attribute classes for each protected attribute
        self.adversary = self._build_adversary()
        self.adversarial_model = self._build_adversarial_model()
        self.prejudice_remover = prejudice_remover
    
    def _build_adversary(self) -> keras.Model:
        intermediate_input = keras.layers.Input(shape=(8,))
        x = keras.layers.Dense(16, activation="relu")(intermediate_input)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(8, activation="relu")(x)

        # for binary classification: 1 output, sigmoid activation
        # for multi-class classification: # classes output, softmax activation
        adversary_outputs = [
            keras.layers.Dense(
                1 if protected_shape == 2 else protected_shape,
                activation="sigmoid" if protected_shape == 2 else "softmax",
                name=f"protected_{protected_name}_prediction")(x)
            for protected_name, protected_shape in zip(self.protected_attribute_names, self.protected_shapes)
        ]
        
        return keras.Model(inputs=intermediate_input, outputs=adversary_outputs)
    
    def _build_adversarial_model(self) -> keras.Model:
        predictor_input = keras.layers.Input(shape=(self.input_shape,))
        predictor_output = self.model(predictor_input)

        # renaming predictor output
        predictor_output = keras.layers.Lambda(lambda x: x, name="adversary_network_output")(predictor_output)

        intermediate_layer = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )(predictor_input)

        adversary_outputs = self.adversary(intermediate_layer)

        adversary_outputs = [
            keras.layers.Lambda(lambda x: x, name=f"protected_{protected_name}_prediction")(output)
            for protected_name, output in zip(self.protected_attribute_names, adversary_outputs)
        ]

        combined_model = keras.Model(inputs=predictor_input, outputs=[predictor_output] + adversary_outputs)
        return combined_model

    def fit(self, X, y, protected_labels):
        optimizer = Adam(learning_rate=self.learning_rate)
 
        protected_labels_reshaped = np.stack(protected_labels, axis=1)

        loss_dict = {
            "adversary_network_output": PrejudiceRemoverLoss(self.lambda_reg, protected_labels_reshaped) if self.prejudice_remover else keras.losses.BinaryCrossentropy(),
            **{f"protected_{protected_name}_prediction": (
                keras.losses.BinaryCrossentropy()
                if protected_shape == 2
                else keras.losses.SparseCategoricalCrossentropy()
            )
            for protected_name, protected_shape in zip(self.protected_attribute_names, self.protected_shapes)
            }
        }

        loss_weights = {
            "adversary_network_output": 1.0,
            **{f"protected_{protected_name}_prediction": self.lambda_reg for protected_name in self.protected_attribute_names}
        }

        self.adversarial_model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights
        )

        adversary_output_names = {f"protected_{protected_name}_prediction": labels for protected_name, labels in zip(self.protected_attribute_names, protected_labels)}
        self.adversarial_model.fit(
            X,
            {"adversary_network_output": y, **adversary_output_names},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        return self
    
    def predict(self, X):
        predictions, *_ = self.adversarial_model.predict(X)
        return (predictions >= 0.5).astype(int).flatten()
    
class MultiAdversary(Vanilla):
    """Extending to multiple adversaries for improved debiasing"""
    
    def __init__(
        self, 
        input_shape, 
        protected_attribute_names,
        protected_shapes,
        num_adversaries=3,
        batch_size=32,
        epochs=20,
        lambda_reg=0.1,
        learning_rate=2e-3,
        prejudice_remover=False
    ):
        super().__init__(input_shape, batch_size, epochs)
        self.protected_attribute_names = protected_attribute_names
        self.protected_shapes = protected_shapes
        self.num_adversaries = num_adversaries
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.prejudice_remover = prejudice_remover
        
        self.predictor = self._build_predictor()
        self.adversaries = [
            self._build_adversary(f"adversary_{i}") 
            for i in range(num_adversaries)
        ]
        
    def _build_predictor(self):
        print("input shape", self.input_shape)
        return keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(self.input_shape,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8, activation="relu", name="intermediate_layer"),
            keras.layers.Dense(1, activation="sigmoid", name="main_output")
        ])
    
    def _build_adversary(self, name):
        """Create a single adversary with unique architecture"""
        # Different adversaries get different architectures
        architectures = {
            'adversary_0': [32, 16, 8],  # Deep network
            'adversary_1': [64, 32],     # Wide network
            'adversary_2': [16, 16, 16]  # Uniform network
        }
        
        # Input layer for intermediate representations
        inputs = keras.layers.Input(shape=(8,))  # Match intermediate_layer size
        x = inputs
        
        # Build hidden layers based on architecture
        hidden_sizes = architectures.get(name, [32, 16])  # Default if more adversaries
        for size in hidden_sizes:
            x = keras.layers.Dense(size, activation="leaky_relu")(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.BatchNormalization()(x)
            
        # Output heads for each protected attribute
        outputs = []
        for attr_name, shape in zip(self.protected_attribute_names, self.protected_shapes):
            output = keras.layers.Dense(
                1 if shape == 2 else shape,
                activation="sigmoid" if shape == 2 else "softmax",
                name=f"{name}_{attr_name}"
            )(x)
            outputs.append(output)
            
        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=name
        )
    
    def fit(self, X, y, protected_labels):
        """Train the model with main task labels and protected attribute labels"""
        # Input layer for the combined model
        input_layer = keras.layers.Input(shape=(self.input_shape,))
        
        # Predictor outputs
        predictor_output = self.predictor(input_layer)
        
        # Extract intermediate output from the predictor for adversaries
        intermediate_layer_model = keras.Model(
            inputs=self.predictor.input,
            outputs=self.predictor.get_layer("intermediate_layer").output
        )
        intermediate_output = intermediate_layer_model(input_layer)
        
        # Adversary outputs
        all_adversary_outputs = {}
        for i, adversary in enumerate(self.adversaries):
            adversary_outputs = adversary(intermediate_output)
            if not isinstance(adversary_outputs, list):
                adversary_outputs = [adversary_outputs]
            
            for attr_name, output in zip(self.protected_attribute_names, adversary_outputs):
                output_name = f"adversary_{i}_{attr_name}"
                all_adversary_outputs[output_name] = output
        
        # Create the combined model
        combined_model_outputs = {"main_output": predictor_output, **all_adversary_outputs}
        combined_model = keras.Model(
            inputs=input_layer,
            outputs=combined_model_outputs
        )
        
        # Compile the model
        loss_dict = {"main_output": "binary_crossentropy"}
        loss_weights = {"main_output": 1.0}
        
        protected_labels_reshaped = np.stack(protected_labels, axis=1)
        # Add adversary loss functions and weights
        for i in range(self.num_adversaries):
            for attr_name, shape in zip(self.protected_attribute_names, self.protected_shapes):
                output_name = f"adversary_{i}_{attr_name}"
                if self.prejudice_remover:
                    loss_dict[output_name] = PrejudiceRemoverLoss(self.lambda_reg, protected_labels_reshaped)
                else:
                    loss_dict[output_name] = (
                        "binary_crossentropy"
                        if shape == 2
                        else "sparse_categorical_crossentropy"
                    )
                loss_weights[output_name] = -self.lambda_reg / self.num_adversaries
        
        combined_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=["accuracy"]
        )
        
        # Prepare training labels
        training_labels = {"main_output": y}
        for i in range(self.num_adversaries):
            for attr_name, labels in zip(self.protected_attribute_names, protected_labels):
                training_labels[f"adversary_{i}_{attr_name}"] = labels
        
        # Train the combined model
        history = combined_model.fit(
            X,
            training_labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        return history
        
    def predict(self, X):
        """Make predictions using the predictor network"""
        predictions = self.predictor.predict(X)
        return (predictions >= 0.5).astype(int).flatten()