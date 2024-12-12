import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from metrics import FairnessMetrics

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

    def __init__(self, input_shape, batch_size=32, epochs=20, learning_rate=3e-3):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        optimizer = keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
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

class GradientReversalLayer(keras.layers.Layer):
    def __init__(self, lambda_reg=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg

    def call(self, x, training=None):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -self.lambda_reg * dy
            return x, grad
        return reverse_gradient(x)

class Adversary(Vanilla):
    """Modifying Vanilla DNN to Adversarial Network"""

    def __init__(self, input_shape, protected_attribute_names, protected_shapes, batch_size=32, epochs=20, lambda_reg=0.1, learning_rate=2e-3):
        super().__init__(input_shape, batch_size, epochs)
        self.lambda_reg = lambda_reg  # lambda in combined loss formula (L = L_{vanilla} - lambda_reg * L_{adversary})
        self.protected_attribute_names = protected_attribute_names
        self.learning_rate = learning_rate
        self.protected_shapes = protected_shapes # list of number of protected attribute classes for each protected attribute
        self.adversary = self._build_adversary()
        self.adversarial_model = self._build_adversarial_model()

    def _build_adversary(self) -> keras.Model:
        intermediate_input = keras.layers.Input(shape=(8,))
        x = GradientReversalLayer(lambda_reg=self.lambda_reg)(intermediate_input)
        x = keras.layers.Dense(16, activation="relu")(x)
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
        optimizer = keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)

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

class ImprovedAdversary(Vanilla):
    """Enhanced adversarial network with dynamic weighting and adaptive learning"""
    
    def __init__(
        self, 
        input_shape, 
        protected_attribute_names,
        protected_shapes,
        batch_size=32,
        epochs=20,
        initial_lambda=0.1,
        max_lambda=2.0,
        learning_rate=2e-3
    ):
        print("protected attribute names:", protected_attribute_names)
        super().__init__(input_shape, batch_size, epochs)
        self.input_shape = input_shape
        self.protected_attribute_names = protected_attribute_names
        self.protected_shapes = protected_shapes
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lambda = initial_lambda
        self.max_lambda = max_lambda
        self.base_learning_rate = learning_rate
        
        self.fairness_history = {name: [] for name in protected_attribute_names}
        self.model = self._build_improved_model()
        
    def _build_improved_model(self):
        
        """Build the complete model ensuring proper tensor connectivity"""
        inputs = keras.layers.Input(shape=(self.input_shape,))
        
        x = keras.layers.Dense(64, activation="leaky_relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(32, activation="leaky_relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(16, activation="leaky_relu")(x)
        x = keras.layers.BatchNormalization()(x)
        
        intermediate = keras.layers.Dense(8, activation="leaky_relu", name="intermediate_layer")(x)
        main_output = keras.layers.Dense(1, activation="sigmoid", name="main_output")(intermediate)
        
        adv_x1 = keras.layers.Dense(32, activation="leaky_relu")(intermediate)
        adv_x1 = keras.layers.BatchNormalization()(adv_x1)
        adv_x1 = keras.layers.Dense(16, activation="leaky_relu")(adv_x1)
        adv_x1 = keras.layers.BatchNormalization()(adv_x1)
        
        adv_x2 = keras.layers.Dense(16, activation="leaky_relu")(intermediate)
        adv_x2 = keras.layers.BatchNormalization()(adv_x2)
        
        adv_x = keras.layers.Add()([adv_x1, adv_x2])
        adv_x = keras.layers.Dropout(0.3)(adv_x)
        
        adversary_outputs = []
        for name, shape in zip(self.protected_attribute_names, self.protected_shapes):
            output = keras.layers.Dense(
                1 if shape == 2 else shape,
                activation="sigmoid" if shape == 2 else "softmax",
                name=f"protected_{name}_prediction"
            )(adv_x)
            adversary_outputs.append(output)
        
        model = keras.Model(inputs=inputs, outputs=[main_output] + adversary_outputs)        
        return model
    
    def fit(self, X, y, protected_labels, validation_data=None):
        """Train the model with dynamic weighting"""
        history = []
        
        for epoch in range(self.epochs):
            current_lr = self._adjust_learning_rate(epoch)
            y_pred = self.predict(X)
            
            metrics = FairnessMetrics(y, y_pred, protected_labels, self.protected_attribute_names)
            fairness_metrics = metrics.statistical_parity_difference()
            lambda_weights = self._update_lambda_weights(fairness_metrics)
            
            self.model.compile(
                optimizer=keras.optimizers.legacy.Adam(learning_rate=current_lr),
                loss={
                    "main_output": "binary_crossentropy",
                    **{f"protected_{name}_prediction": (
                        "binary_crossentropy"
                        if shape == 2
                        else "sparse_categorical_crossentropy"
                    )
                    for name, shape in zip(
                        self.protected_attribute_names, self.protected_shapes)}
                },
                loss_weights={
                    "main_output": 1.0,
                    **{f"protected_{name}_prediction": -weight 
                       for name, weight in lambda_weights.items()}
                }
            )
            
            training_data = {
                "main_output": y,
                **{f"protected_{name}_prediction": labels 
                   for name, labels in zip(
                    self.protected_attribute_names, protected_labels)}
            }
            
            epoch_history = self.model.fit(
                X,
                training_data,
                batch_size=self.batch_size,
                epochs=1,
                verbose=1,
                validation_data=validation_data
            )
            
            history.append({
                'epoch': epoch,
                'learning_rate': current_lr,
                'lambda_weights': lambda_weights,
                'fairness_metrics': fairness_metrics,
                **epoch_history.history
            })
            
        return history
    
    def predict(self, X):
        """Make predictions using the main output"""
        predictions = self.model.predict(X)[0]  # Get main output
        return (predictions >= 0.5).astype(int).flatten()
    
    def _adjust_learning_rate(self, epoch):
        """Implement adaptive learning rate schedule"""
        # Warm-up phase
        if epoch < self.epochs * 0.3:  
            return self.base_learning_rate
        # Normal training
        elif epoch < self.epochs * 0.6:  
            return self.base_learning_rate * 0.1
         # Fine-tuning
        else: 
            return self.base_learning_rate * 0.01
    
    def _update_lambda_weights(self, fairness_metrics):
        """Dynamically adjust lambda weights based on fairness metrics"""
        weights = {}
        print("fairness metrics: ", fairness_metrics)
        for name in self.protected_attribute_names:
            metric = fairness_metrics.get(f'spd_{name}', 0)
            self.fairness_history[name].append(metric)
            
            if len(self.fairness_history[name]) > 1:
                trend = self.fairness_history[name][-1] - self.fairness_history[name][-2]
                if trend > 0 or abs(metric) > 0.1:
                    weight = min(self.max_lambda, self.initial_lambda * (1 + abs(metric) * 10))
                else:
                    weight = self.initial_lambda
            else:
                weight = self.initial_lambda
            
            weights[name] = weight
        return weights

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
        learning_rate=2e-3
    ):
        super().__init__(input_shape, batch_size, epochs)
        self.protected_attribute_names = protected_attribute_names
        self.protected_shapes = protected_shapes
        self.num_adversaries = num_adversaries
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate

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
            'adversary_0': [64, 32, 16],  # Deep network
            'adversary_1': [128, 64],     # Wide network
            'adversary_2': [32, 32, 32]  # Uniform network
        }

        inputs = keras.layers.Input(shape=(8,))
        x = inputs

        hidden_sizes = architectures.get(name, [32, 16])
        for size in hidden_sizes:
            x = keras.layers.Dense(size, activation="leaky_relu")(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.BatchNormalization()(x)

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
        input_layer = keras.layers.Input(shape=(self.input_shape,))
        predictor_output = self.predictor(input_layer)

        intermediate_layer_model = keras.Model(
            inputs=self.predictor.input,
            outputs=self.predictor.get_layer("intermediate_layer").output
        )
        intermediate_output = intermediate_layer_model(input_layer)

        all_adversary_outputs = {}
        for i, adversary in enumerate(self.adversaries):
            adversary_outputs = adversary(intermediate_output)
            if not isinstance(adversary_outputs, list):
                adversary_outputs = [adversary_outputs]

            for attr_name, output in zip(self.protected_attribute_names, adversary_outputs):
                output_name = f"adversary_{i}_{attr_name}"
                all_adversary_outputs[output_name] = output

        combined_model_outputs = {"main_output": predictor_output, **all_adversary_outputs}
        combined_model = keras.Model(
            inputs=input_layer,
            outputs=combined_model_outputs
        )

        loss_dict = {"main_output": "binary_crossentropy"}
        loss_weights = {"main_output": 1.0}

        for i in range(self.num_adversaries):
            for attr_name, shape in zip(self.protected_attribute_names, self.protected_shapes):
                output_name = f"adversary_{i}_{attr_name}"
                loss_dict[output_name] = (
                    "binary_crossentropy"
                    if shape == 2
                    else "sparse_categorical_crossentropy"
                )
                loss_weights[output_name] = -self.lambda_reg / self.num_adversaries

        combined_model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=["accuracy"]
        )

        training_labels = {"main_output": y}
        for i in range(self.num_adversaries):
            for attr_name, labels in zip(self.protected_attribute_names, protected_labels):
                training_labels[f"adversary_{i}_{attr_name}"] = labels

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