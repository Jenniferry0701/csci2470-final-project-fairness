from metrics import compute_all_metrics
from utils import read_and_split_data
from model import Vanilla
from sklearn.preprocessing import StandardScaler

dataset = "compas"
compas_path = f"./data/{dataset}_processed.csv"
protected_attributes = ['sex', 'race']
output_file = f"./results/{dataset}_fairness.csv"
X_train, X_test, y_train, y_test = read_and_split_data(compas_path, "Probability")

print(f"Class distribution in training set: {y_train.value_counts()}")
print(f"Class distribution in test set: {y_test.value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Vanilla(input_shape=X_train_scaled.shape[1], epochs=10)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

metrics = compute_all_metrics(y_true=y_test.values, y_pred=y_pred, protected_attributes=X_test[protected_attributes[0]].values)
print(metrics)