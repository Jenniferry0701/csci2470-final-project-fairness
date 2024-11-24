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
            keras.layers.Dense(1, activation="sigmoid")
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