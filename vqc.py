import pandas as pd
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_machine_learning.algorithms import VQC


def callback_graph(weights, obj_func_eval):
    print(f"Objective function value: {obj_func_eval}")


if __name__ == '__main__':
    df = pd.read_csv('datasets/vlds.csv', index_col=0)
    feature_dimension = df.shape[1] - 1
    print(f"Feature dimension: {feature_dimension}")

    train = df.sample(frac=0.75, random_state=42)
    test = df.drop(train.index)

    # Separate the features from the labels
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('label').to_numpy()
    test_labels = test_features.pop('label').to_numpy()

    algorithm_globals.random_seed = 42
    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(backend, shots=1024)

    vqc = VQC(
        num_qubits=feature_dimension,
        quantum_instance=quantum_instance,
        optimizer=COBYLA(maxiter=200),
        callback=callback_graph,
        feature_map=ZZFeatureMap(feature_dimension=feature_dimension, reps=1, entanglement='sca'),
        # ansatz=TwoLocal(feature_dimension, ['ry', 'rz'], 'cz', reps=3),
    )

    vqc.fit(train_features, train_labels)

    print(f"Test accuracy: {vqc.score(test_features, test_labels)}")
