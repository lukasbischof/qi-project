import concurrent
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import KFold
from torch.autograd import Function
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import pandas as pd


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x_train = torch.tensor(np.array(x, dtype='float32'), dtype=torch.float32)
        self.y_train = torch.tensor(np.array(y, dtype='int8'), dtype=torch.int64)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class Net(nn.Module):
    def __init__(self, feature_dimension, backend):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_dimension, 16)
        self.fc2 = nn.Linear(16, 1)
        self.hybrid = Hybrid(backend, shots=100, shift=np.pi / 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


class HybridNNModel:
    def __init__(self, model, fold, dataset_name, loss_func=nn.NLLLoss, optimizer=None, epochs=20):
        self._start_time = None
        self.model = model
        self.loss_list = []
        self.loss_func = loss_func()
        self.execution_time = 0  # in seconds
        self.fold = fold
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01) if not optimizer else optimizer
        self.loaded = False

    def fit(self, train_loader):
        self.model.train()
        self._start_time = time.time()
        for epoch in range(self.epochs):
            total_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model(data)
                # Calculating loss
                loss = self.loss_func(output, target)
                # Backward pass
                loss.backward()
                # Optimize the weights
                self.optimizer.step()

                total_loss.append(loss.item())
            self.loss_list.append(sum(total_loss) / len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / self.epochs, self.loss_list[-1]))
        self.execution_time = time.time() - self._start_time
        return self

    def score(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                output = self.model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            return correct / len(test_loader) * 100

    def save(self):
        torch.save(torch.jit.export(self.model), f'classifiers/hybrid_nn/{self.save_file_name()}')

    def save_file_name(self):
        opt = self.optimizer.__class__.__name__
        lf = self.loss_func.__class__.__name__
        return f'{self.dataset_name}_fold_{self.fold}_{self.epochs}_epochs_{opt}_{lf}.pt'

    def load(self):
        path = f'classifiers/hybrid_nn/{self.save_file_name()}'
        if not os.path.exists(path):
            return False

        loaded = torch.load(path)
        self.model = loaded
        self.loaded = True
        return True


if __name__ == '__main__':
    # Configurations
    dataset_name = 'vlds_1k'
    folds = 10
    epochs = 100
    loss_func = nn.NLLLoss

    # Setup Quantum Backend
    simulator = qiskit.Aer.get_backend('aer_simulator')

    circuit = QuantumCircuit(1, simulator, 100)
    print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
    circuit._circuit.draw(output='mpl')

    # Prepare Data
    df = pd.read_csv(f'datasets/{dataset_name}.csv', index_col=0)
    feature_dimension = df.shape[1] - 1

    all_features = df.copy()
    all_labels = all_features.pop('label')

    train = df.sample(frac=0.75, random_state=42)
    test = df.drop(train.index)

    # Separate the features from the labels
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('label')
    test_labels = test_features.pop('label')

    test_loader = torch.utils.data.DataLoader(MyDataset(test_features, test_labels), batch_size=1, shuffle=True)

    # Train and test the model
    models = [
        HybridNNModel(
            Net(feature_dimension=feature_dimension, backend=simulator),
            fold=fold,
            loss_func=loss_func,
            epochs=epochs,
            dataset_name=dataset_name
        ) for fold in range(folds)
    ]

    for model in models:
        if model.load():
            print(f'Loaded model {model.save_file_name()}')

    if not all(model.loaded for model in models):
        k_fold = KFold(n_splits=10)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i, (train_index, test_index) in enumerate(k_fold.split(train_features)):
                model = models[i]
                if model.loaded:
                    continue

                current_train_features = train_features.iloc[train_index]
                current_train_labels = train_labels.iloc[train_index]
                train_loader = torch.utils.data.DataLoader(MyDataset(current_train_features, current_train_labels),
                                                           batch_size=1, shuffle=True)
                futures.append(executor.submit(model.fit, train_loader))
            executor.shutdown(wait=True)
            models = [future.result() for future in futures]

        fig, axs = plt.subplots(2, folds // 2, figsize=(folds * 2, 2 * 2))
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Neg Log Likelihood Loss')
        for i, model in enumerate(models):
            ax = axs[i // (folds // 2), i % (folds // 2)]
            ax.plot(model.loss_list, label=f'Fold {i}')
            ax.title.set_text(f'Fold {i}')
            ax.grid()
        plt.show()

    accuracies = np.array([model.score(test_loader) for model in models])
    for i, model in enumerate(models):
        if not model.loaded:
            model.save()

        print('Fold %d accuracy %.2f%% on test set, trained in %.5f seconds' % (
            i, accuracies[i], model.execution_time
        ))
    print(f"Accuracy mean: {accuracies.mean()}%, std: {accuracies.std()}")
