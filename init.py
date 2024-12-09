from metrics import compute_all_metrics
from utils import read_and_split_data, get_args
from model import Vanilla, Adversary, MultiAdversary
from sklearn.preprocessing import StandardScaler
import pandas as pd

def initialize_protected_attributes(dataset):
    """
    Initialize protected attributes based on the input dataset.
    """
    if dataset in ["default"]:
        return ['sex', 'age']
    else:
        return ['sex', 'race']
    
def initialize_non_default_args(args):
    if not args.dataset_path:
        args.dataset_path = f"./data/{args.dataset}_processed.csv"
    if not args.output_file:
        args.output_file = f"./results/{args.dataset}_fairness.csv"
    if not args.protected_attributes:
        args.protected_attributes = initialize_protected_attributes(args.dataset)

def preprocess_data(dataset_path):
    X_train, X_test, y_train, y_test = read_and_split_data(dataset_path, "Probability")
    print(f"Class distribution in training set: {y_train.value_counts()}")
    print(f"Class distribution in test set: {y_test.value_counts()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def extract_protected_attributes(X_train, X_test, protected_attributes):
    protected_train = [X_train[attr].values for attr in protected_attributes]
    protected_test = [X_test[attr].values for attr in protected_attributes]
    return protected_train, protected_test

def train_vanilla_model(X_train_scaled, y_train, num_epochs):
    model = Vanilla(input_shape=X_train_scaled.shape[1], epochs=num_epochs)
    model.fit(X_train_scaled, y_train)
    return model

def train_adversarial_model(X_train_scaled, y_train, protected_attribute_names, protected_shapes, num_epochs, lambda_reg, protected_train, learning_rate, prejudice_remover):
    adv_model = Adversary(
        input_shape=X_train_scaled.shape[1],
        protected_attribute_names=protected_attribute_names,
        protected_shapes=protected_shapes,
        epochs=num_epochs,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        prejudice_remover=prejudice_remover
    )
    adv_model.fit(X_train_scaled, y_train, protected_train)
    return adv_model

def train_multi_adversarial_model(X_train_scaled, y_train, protected_attribute_names, protected_shapes, num_epochs, lambda_reg, protected_train, learning_rate, num_adversaries, prejudice_remover):
    print("protected attributers:", protected_attribute_names)
    print("protected shapes:", protected_shapes)
    multi_adv_model = MultiAdversary(
        input_shape=X_train_scaled.shape[1],
        num_adversaries=num_adversaries,
        epochs = num_epochs,
        protected_attribute_names = protected_attribute_names,
        protected_shapes = protected_shapes,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        prejudice_remover=prejudice_remover
    )
    multi_adv_model.fit(X_train_scaled, y_train, protected_train)
    return multi_adv_model

def evaluate_model(model, X_test_scaled, y_test, protected_attributes):
    y_pred = model.predict(X_test_scaled)
    metrics = compute_all_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        protected_attributes=protected_attributes
    )
    return metrics

def preprocess_results_for_output(results):
    # Flatten the nested `metrics` dictionary for each result
    processed_results = []
    for result in results:
        flattened_result = result.copy()
        flattened_result.update(result.pop('metrics', {}))  # Merge metrics into the main dictionary
        processed_results.append(flattened_result)
    return processed_results

def save_results_to_output(results, output_file):
    results = pd.DataFrame(results)
    results.round(2).to_csv(output_file)

def compare_lambda_reg_values_on_adversarial_model():
    lambda_reg_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    all_results = []

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(args.dataset_path)
    protected_train, protected_test = extract_protected_attributes(X_train, X_test, args.protected_attributes)
    protected_shapes = [len(pd.Series(attr).unique()) for attr in protected_train]

    # Iterate over different lambda_reg values
    for lambda_reg in lambda_reg_values:
        if lambda_reg == 0.01 and args.dataset == 'adult':
            continue
        print(f"Training Adversarial model with lambda_reg = {lambda_reg}")
        
        # Train Adversarial DNN model
        adv_model = train_adversarial_model(
            X_train_scaled=X_train_scaled, 
            y_train=y_train, 
            protected_attribute_names=args.protected_attributes, 
            protected_shapes=protected_shapes, 
            num_epochs=args.epochs, 
            lambda_reg=lambda_reg, 
            protected_train=protected_train,
            learning_rate=args.learning_rate,
            prejudice_remover=False
        )
        adv_metrics = evaluate_model(
            adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
        )
        print(f"Adversarial Metrics (lambda_reg={lambda_reg}): ", adv_metrics)
        
        # Append results
        all_results.append({
            "lambda_reg": lambda_reg,
            "model_type": "Adversary",
            "metrics": adv_metrics
        })
        
        # Train Adversarial model with Prejudice Remover
        adv_model_with_prejudice_remover = train_adversarial_model(
            X_train_scaled=X_train_scaled, 
            y_train=y_train, 
            protected_attribute_names=args.protected_attributes, 
            protected_shapes=protected_shapes, 
            num_epochs=args.epochs, 
            lambda_reg=lambda_reg, 
            protected_train=protected_train,
            learning_rate=args.learning_rate,
            prejudice_remover=True
        )
        adv_metrics_with_prejudice_remover = evaluate_model(
            adv_model_with_prejudice_remover, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
        )
        print(f"Adversarial Metrics with Prejudice Remover (lambda_reg={lambda_reg}): ", adv_metrics_with_prejudice_remover)
        
        # Append results
        all_results.append({
            "lambda_reg": lambda_reg,
            "model_type": "Adversary with Prejudice Remover",
            "metrics": adv_metrics_with_prejudice_remover
        })

    # Save all results to output
    save_results_to_output(preprocess_results_for_output(all_results), args.output_file)

def compare_lambda_reg_values_on_multi_adversarial_model():
    lambda_reg_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    all_results = []

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(args.dataset_path)
    protected_train, protected_test = extract_protected_attributes(X_train, X_test, args.protected_attributes)
    protected_shapes = [len(pd.Series(attr).unique()) for attr in protected_train]

    # Iterate over different lambda_reg values
    for lambda_reg in lambda_reg_values:
        print(f"Training MultiAdversary model with lambda_reg = {lambda_reg}")

        multi_adv_model = train_multi_adversarial_model(
            X_train_scaled=X_train_scaled, 
            y_train=y_train, 
            protected_attribute_names=args.protected_attributes, 
            protected_shapes=protected_shapes, 
            num_epochs=args.epochs, 
            lambda_reg=args.lambda_reg, 
            protected_train=protected_train,
            learning_rate=1e-5,
            num_adversaries=2,
            prejudice_remover=False
        )
        multi_adv_metrics = evaluate_model(
            multi_adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
        )
        print(f"Multi Adversarial Metrics (lambda_reg={lambda_reg}): ", multi_adv_metrics)

        all_results.append({
            "lambda_reg": lambda_reg,
            "model_type": "MultiAdversary",
            "metrics": multi_adv_metrics
        })

        multi_adv_model_with_prejudice_remover = train_multi_adversarial_model(
            X_train_scaled=X_train_scaled, 
            y_train=y_train, 
            protected_attribute_names=args.protected_attributes, 
            protected_shapes=protected_shapes, 
            num_epochs=args.epochs, 
            lambda_reg=args.lambda_reg, 
            protected_train=protected_train,
            learning_rate=1e-5,
            num_adversaries=2,
            prejudice_remover=True
        )
        multi_adv_metrics_with_prejudice_remover = evaluate_model(
            multi_adv_model_with_prejudice_remover, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
        )
        print(f"Multi Adversarial Metrics with Prejudice Remover (lambda_reg={lambda_reg}): ", multi_adv_metrics_with_prejudice_remover)
        
        all_results.append({
            "lambda_reg": lambda_reg,
            "model_type": "MultiAdversary",
            "metrics": multi_adv_metrics_with_prejudice_remover
        })

    # Save all results to output
    save_results_to_output(preprocess_results_for_output(all_results), args.output_file)

def evaluate_models():
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(args.dataset_path)
    protected_train, protected_test = extract_protected_attributes(X_train, X_test, args.protected_attributes)
    protected_shapes = [len(pd.Series(attr).unique()) for attr in protected_train]

    # Train Vanilla DNN model
    vanilla_model = train_vanilla_model(X_train_scaled, y_train, args.epochs)
    vanilla_metrics = evaluate_model(model=vanilla_model, 
                                     X_test_scaled=X_test_scaled, 
                                     y_test=y_test, 
                                     protected_attributes=[X_test[attr].values for attr in args.protected_attributes]
    )
    print("vanilla: ",vanilla_metrics)

    # Train Adversarial DNN model
    adv_model = train_adversarial_model(X_train_scaled=X_train_scaled, 
                                        y_train=y_train, 
                                        protected_attribute_names=args.protected_attributes, 
                                        protected_shapes=protected_shapes, 
                                        num_epochs=args.epochs, 
                                        lambda_reg=args.lambda_reg, 
                                        protected_train=protected_train,
                                        learning_rate=args.learning_rate,
                                        prejudice_remover=False)
    adv_metrics = evaluate_model(
        adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
    )
    print("adversary: ", adv_metrics)

    adv_model_with_prejudice_remover = train_adversarial_model(X_train_scaled=X_train_scaled, 
                                                               y_train=y_train, 
                                                               protected_attribute_names=args.protected_attributes, 
                                                               protected_shapes=protected_shapes, 
                                                               num_epochs=args.epochs, 
                                                               lambda_reg=args.lambda_reg, 
                                                               protected_train=protected_train,
                                                               learning_rate=args.learning_rate,
                                                               prejudice_remover=True)
    adv_metrics_with_prejudice_remover = evaluate_model(
        adv_model_with_prejudice_remover, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
    )
    print("adversary with prejudice remover: ", adv_metrics_with_prejudice_remover)

    multi_adv_model = train_multi_adversarial_model(
        X_train_scaled=X_train_scaled, 
        y_train=y_train, 
        protected_attribute_names=args.protected_attributes, 
        protected_shapes=protected_shapes, 
        num_epochs=args.epochs, 
        lambda_reg=args.lambda_reg, 
        protected_train=protected_train,
        learning_rate=1e-5,
        num_adversaries=2,
        prejudice_remover=False
    )
    multi_adv_metrics = evaluate_model(
        multi_adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
    )

    print("multi: ", multi_adv_metrics)

    multi_adv_model_with_prejudice_remover = train_multi_adversarial_model(
        X_train_scaled=X_train_scaled, 
        y_train=y_train, 
        protected_attribute_names=args.protected_attributes, 
        protected_shapes=protected_shapes, 
        num_epochs=args.epochs, 
        lambda_reg=args.lambda_reg, 
        protected_train=protected_train,
        learning_rate=1e-5,
        num_adversaries=2,
        prejudice_remover=True
    )
    multi_adv_metrics_with_prejudice_remover = evaluate_model(
        multi_adv_model_with_prejudice_remover, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
    )

    print("multi with prejudice remover: ", multi_adv_metrics_with_prejudice_remover)
    save_results_to_output([vanilla_metrics, adv_metrics, adv_metrics_with_prejudice_remover, multi_adv_metrics], args.output_file)

if __name__ == "__main__":
    arg_list = [
        ('-d', '--dataset', 'compas', str),
        ('-dp', '--dataset_path', None, str),
        ('-pa', '--protected_attributes', None, list),
        ('-ts', '--target_string', 'Probability', str),
        ('-o', '--output_file', None, str),
        ('-e', '--epochs', 10, int),
        ('-lreg', '--lambda_reg', 0.1, float),
        ('-lr', '--learning-rate', 3e-3, float),
        ('-tlreg', '--test_lambda_reg', True, bool)
    ]
    args = get_args(arg_list)
    initialize_non_default_args(args)
    evaluate_models()
    if args.test_lambda_reg:
        args.output_file = f"./results/{args.dataset}_fairness_lambda_reg_value_test.csv"
        compare_lambda_reg_values_on_adversarial_model()
        compare_lambda_reg_values_on_multi_adversarial_model()
   
    
