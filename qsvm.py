import pandas as pd
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.svm import SVC

if __name__ == '__main__':
    df = pd.read_csv('datasets/vlds.csv', index_col=0)

    # Split the data into training and test sets
    train = df.sample(frac=0.75, random_state=42)
    test = df.drop(train.index)

    # Separate the features from the labels
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('label')
    test_labels = test_features.pop('label')

    # Setup circuit
    algorithm_globals.random_seed = 42
    quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=1024)

    # Setup QSVM
    feature_map = ZZFeatureMap(feature_dimension=train_features.shape[1], reps=2)
    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    svc = SVC(kernel=kernel.evaluate)
    svc.fit(train_features, train_labels)
    score = svc.score(test_features, test_labels)
    print(f"QSVM score: {score}")
