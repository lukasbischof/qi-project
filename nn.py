import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    # Neural Network on VLDS data set

    df = pd.read_csv('datasets/custom_dataset_10k.csv', index_col=0)
    feature_dimension = df.shape[1] - 1

    print(f"Feature dimension: {feature_dimension}")

    train = df.sample(frac=0.75, random_state=42)
    test = df.drop(train.index)

    # Separate the features from the labels
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('label')
    test_labels = test_features.pop('label')

    models = []
    k_fold = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(k_fold.split(train_features)):
        print(f"==> Fold {i+1}/{k_fold.get_n_splits()}")

        current_train_features = train_features.iloc[train_index].to_numpy()
        current_train_labels = train_labels.iloc[train_index].to_numpy()

        # Setup Neural Network
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[feature_dimension]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        model.fit(train_features, train_labels, epochs=10)
        models.append(model)

    accuracies = np.array([model.evaluate(test_features, test_labels)[1] for model in models])
    print(f"Accuracies: {accuracies}")
    print(f"Average accuracy: {accuracies.mean()}, std: {accuracies.std()}")
    # test_loss, test_acc = model.evaluate(test_features, test_labels)
    # print(f"Test accuracy: {test_acc}")
    # print(f"Test loss: {test_loss}")
