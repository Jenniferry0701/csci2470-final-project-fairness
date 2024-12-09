from metrics import compute_all_metrics
from utils import read_data, read_and_split_data, get_args
from model import Vanilla, Adversary, MultiAdversary
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold


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

def train_adversarial_model(X_train_scaled, y_train, protected_attribute_names, protected_shapes, num_epochs, lambda_reg, protected_train, learning_rate):
    adv_model = Adversary(
        input_shape=X_train_scaled.shape[1],
        protected_attribute_names=protected_attribute_names,
        protected_shapes=protected_shapes,
        epochs=num_epochs,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate
    )
    adv_model.fit(X_train_scaled, y_train, protected_train)
    return adv_model

def train_multi_adversarial_model(X_train_scaled, y_train, protected_attribute_names, protected_shapes, num_epochs, lambda_reg, protected_train, learning_rate, num_adversaries):
    print("protected attributers:", protected_attribute_names)
    print("protected shapes:", protected_shapes)
    multi_adv_model = MultiAdversary(
        input_shape=X_train.shape[1],
        num_adversaries=num_adversaries,
        epochs = num_epochs,
        protected_attribute_names = protected_attribute_names,
        protected_shapes = protected_shapes,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate
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

def save_results_to_output(results, output_file):
    results = pd.DataFrame(results)
    results.round(2).to_csv(output_file)
    
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
        ('-f', '--folds', 5, int)
    ]
    args = get_args(arg_list)
    initialize_non_default_args(args)
    X, y = read_data(args.dataset_path, "Probability")
    # learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    learning_rates=[3e-3]

    fold_metrics = []
    for lr in learning_rates:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            protected_train, protected_test = extract_protected_attributes(X_train, X_test, args.protected_attributes)
            protected_shapes = [len(pd.Series(attr).unique()) for attr in protected_train]

            vanilla_model = train_vanilla_model(X_train_scaled, y_train, args.epochs)
            vanilla_metrics = evaluate_model(model=vanilla_model, 
                                            X_test_scaled=X_test_scaled, 
                                            y_test=y_test, 
                                            protected_attributes=[X_test[attr].values for attr in args.protected_attributes],
             )
            vanilla_metrics["fold"] = fold
            vanilla_metrics["model"] = "vanilla"
            vanilla_metrics["lr"] = str(lr)
            fold_metrics.append(vanilla_metrics)

            adv_model = train_adversarial_model(X_train_scaled=X_train_scaled, 
                                                y_train=y_train, 
                                                protected_attribute_names=args.protected_attributes, 
                                                protected_shapes=protected_shapes, 
                                                num_epochs=args.epochs, 
                                                lambda_reg=args.lambda_reg, 
                                                protected_train=protected_train,
                                                learning_rate=lr)
            adv_metrics = evaluate_model(
                adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
            )
            adv_metrics["fold"] = fold
            adv_metrics["model"] = "adversary"
            adv_metrics["lr"] = str(lr)
            adv_metrics["lambda_reg"] = args.lambda_reg
            fold_metrics.append(adv_metrics)


            # multi_adv_model = train_multi_adversarial_model(
            #     X_train_scaled=X_train_scaled, 
            #     y_train=y_train, 
            #     protected_attribute_names=args.protected_attributes, 
            #     protected_shapes=protected_shapes, 
            #     num_epochs=args.epochs, 
            #     lambda_reg=args.lambda_reg, 
            #     protected_train=protected_train,
            #     learning_rate=lr,
            #     num_adversaries = 2
            # )
            
            # multi_adv_metrics = evaluate_model(
            #     multi_adv_model, X_test_scaled, y_test, [X_test[attr].values for attr in args.protected_attributes]
            # )
            # multi_adv_metrics["fold"] = fold
            # multi_adv_metrics["model"] = "multi-adversary"
            # multi_adv_metrics["lr"] = lr
            # fold_metrics.append(multi_adv_metrics)


    # print("multi: ", multi_adv_metrics)
        save_results_to_output(fold_metrics, args.output_file)


    




