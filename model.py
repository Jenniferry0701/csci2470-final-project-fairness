import tensorflow as tf
from tensorflow import keras
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
            optimizer="nadam",
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

    def __init__(self, input_shape, protected_attribute_names, protected_shapes, batch_size=32, epochs=20, lambda_reg=0.1, lr=2e-3):
        super().__init__(input_shape, batch_size, epochs)
        self.lambda_reg = lambda_reg  # lambda in combined loss formula (L = L_{vanilla} - lambda_reg * L_{adversary})
        self.protected_attribute_names = protected_attribute_names
        self.lr = lr
        self.protected_shapes = protected_shapes # list of number of protected attribute classes for each protected attribute
        self.adversary = self._build_adversary()
        self.adversarial_model = self._build_adversarial_model()
    
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
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        
        self.adversarial_model.compile(
            optimizer=optimizer,
            loss={
                "adversary_network_output": keras.losses.BinaryCrossentropy(),
                **{f"protected_{protected_name}_prediction": (
                   keras.losses.BinaryCrossentropy()
                   if protected_shape == 2
                   else keras.losses.SparseCategoricalCrossentropy()
                )
                for protected_name, protected_shape in zip(self.protected_attribute_names, self.protected_shapes)
                }
            },
            loss_weights={
                "adversary_network_output": 1.0,
                **{f"protected_{protected_name}_prediction": self.lambda_reg for protected_name in self.protected_attribute_names}
            }
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