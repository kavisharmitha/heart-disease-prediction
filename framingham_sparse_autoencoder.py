import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# ============================================================
# Load Preprocessed Training and Testing Data
# ============================================================

def load_data():

    results_path = Path(__file__).resolve().parent.parent / "results"

    X_train = pd.read_csv(results_path / "framingham_X_train.csv")
    X_test = pd.read_csv(results_path / "framingham_X_test.csv")

    return X_train, X_test


# ============================================================
# Build Sparse Autoencoder
# ============================================================

def build_autoencoder(input_dim):

    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(

        6,

        activation="relu",

        activity_regularizer=regularizers.l1(1e-4)

    )(input_layer)

    # Decoder
    decoded = Dense(

        input_dim,

        activation="linear"

    )(encoded)

    autoencoder = Model(

        inputs=input_layer,

        outputs=decoded

    )

    encoder = Model(

        inputs=input_layer,

        outputs=encoded

    )

    autoencoder.compile(

        optimizer="adam",

        loss="mse"

    )

    return autoencoder, encoder


# ============================================================
# Train Sparse Autoencoder
# ============================================================

def train_autoencoder(autoencoder, X_train):

    autoencoder.fit(

        X_train,

        X_train,

        epochs=100,

        batch_size=16,

        validation_split=0.20,

        shuffle=True,

        verbose=1

    )


# ============================================================
# Extract Encoded Features
# ============================================================

def extract_features(encoder, X_train, X_test):

    train_features = encoder.predict(X_train)

    test_features = encoder.predict(X_test)

    return train_features, test_features


# ============================================================
# Save Encoded Features
# ============================================================

def save_features(train_features, test_features):

    results_path = Path(__file__).resolve().parent.parent / "results"

    pd.DataFrame(train_features).to_csv(

        results_path / "framingham_encoded_train.csv",

        index=False

    )

    pd.DataFrame(test_features).to_csv(

        results_path / "framingham_encoded_test.csv",

        index=False

    )

    print("\nEncoded Features Saved Successfully.")


# ============================================================
# Save Model
# ============================================================

def save_model(autoencoder):

    models_path = Path(__file__).resolve().parent.parent / "models"

    models_path.mkdir(exist_ok=True)

    autoencoder.save(

        models_path / "framingham_sparse_autoencoder.keras"

    )

    print("Sparse Autoencoder Saved Successfully.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FRAMINGHAM SPARSE AUTOENCODER")
    print("=" * 70)

    X_train, X_test = load_data()

    input_dimension = X_train.shape[1]

    print(f"\nInput Features : {input_dimension}")

    autoencoder, encoder = build_autoencoder(

        input_dimension

    )

    train_autoencoder(

        autoencoder,

        X_train

    )

    encoded_train, encoded_test = extract_features(

        encoder,

        X_train,

        X_test

    )

    save_features(

        encoded_train,

        encoded_test

    )

    save_model(

        autoencoder

    )

    print("\nSparse Autoencoder Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()