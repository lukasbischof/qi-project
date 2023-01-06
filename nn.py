import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    # Neural Network on VLDS data set

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

    test_loss, test_acc = model.evaluate(test_features, test_labels)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
