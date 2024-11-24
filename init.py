from metrics import compute_all_metrics
from utils import read_and_split_data, get_args
from model import Vanilla
from sklearn.preprocessing import StandardScaler

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

def train_model(X_train_scaled, y_train, num_epochs):
    model = Vanilla(input_shape=X_train_scaled.shape[1], epochs=num_epochs)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test, protected_attributes):
    y_pred = model.predict(X_test_scaled)
    metrics = compute_all_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        protected_attributes=protected_attributes
    )
    return metrics

if __name__ == "__main__":
    arg_list = [
        ('-d', '--dataset', 'compas', str),
        ('-dp', '--dataset_path', None, str),
        ('-pa', '--protected_attributes', None, list),
        ('-ts', '--target_string', 'Probability', str),
        ('-o', '--output_file', None, str),
        ('-e', '--epochs', 10, int),
    ]
    args = get_args(arg_list)
    initialize_non_default_args(args)
   
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(args.dataset_path)
    model = train_model(X_train_scaled, y_train, args.epochs)
    metrics = evaluate_model(model, X_test_scaled, y_test, X_test[args.protected_attributes[0]].values)
    print(metrics)
    





