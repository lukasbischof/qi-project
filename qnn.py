# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import pandas as pd
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.metrics import ConfusionMatrixDisplay

# Configuration
RANDOM_SEED = 42
SHOTS = 1024
FEATURES = 4
OUTPUT_SHAPE = 2
ENTANGLEMENT = "linear"
TRAIN_DATA_SPLIT = 0.75
objective_func_vals = []


def parity(x):
    return "{:b}".format(x).count("1") % 2


if __name__ == '__main__':
    df = pd.read_csv('datasets/vlds.csv', index_col=0)
    df.count()

    print(f'Rows with label = 1: #{df.where(df["label"] == 1.0)["label"].count()}')
    print(f'Rows with label = 0: #{df.where(df["label"] == 0.0)["label"].count()}')

    # Split the data into training and test sets
    train = df.sample(frac=TRAIN_DATA_SPLIT, random_state=RANDOM_SEED)
    test = df.drop(train.index)

    # Separate the features from the labels
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('label')
    test_labels = test_features.pop('label')

    # Setup circuit
    algorithm_globals.random_seed = RANDOM_SEED
    quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=SHOTS)

    feature_map = ZZFeatureMap(feature_dimension=FEATURES, reps=2)
    ansatz = RealAmplitudes(FEATURES, entanglement=ENTANGLEMENT, reps=1)
    qc = QuantumCircuit(FEATURES)
    qc.append(feature_map, range(FEATURES))
    qc.append(ansatz, range(FEATURES))
    # qc.decompose().draw(output='mpl')
    # plt.show()

    # Setup QNN
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        output_shape=OUTPUT_SHAPE,
        interpret=parity,
    )


    def callback_graph(weights, obj_func_eval):
        print(f'Objective function value: {obj_func_eval}')


    sampler_classifier = NeuralNetworkClassifier(
        neural_network=sampler_qnn,
        optimizer=COBYLA(maxiter=30),
        callback=callback_graph
    )

    # Train the model
    plt.rcParams["figure.figsize"] = (12, 6)
    sampler_classifier.fit(train_features, train_labels)

    if not os.path.exists('classifiers'):
        os.makedirs('classifiers')

    sampler_classifier.save('classifiers/sampler_classifier.joblib')

    y_predict = sampler_classifier.predict(test_features)
    accuracy = sum(y_predict == test_labels) / len(test_labels)
    print(f"Accuracy: {accuracy * 100}%")

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        y_predict,
        display_labels=["0", "1"]
    ).plot()
    plt.show()
