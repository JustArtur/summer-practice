import json
import os
from typing import Dict

import joblib
from sklearn.metrics import accuracy_score

from src.data_loader import get_datasets, ensure_directory_exists
from src.model import create_mnist_model

DATA_DIR = "./data"
MODELS_DIR = "./models"
EPOCHS = 1
TRAIN_SUBSET = 1000  # Маленький датасет для быстрого прогона CI
TEST_SUBSET = 200    # Маленький датасет для быстрого прогона CI
RANDOM_STATE = 42


def train() -> Dict[str, float]:
    ensure_directory_exists(MODELS_DIR)

    x_train, y_train, x_test, y_test = get_datasets(
        dataset_directory=DATA_DIR,
        train_subset=TRAIN_SUBSET,
        test_subset=TEST_SUBSET,
        random_state=RANDOM_STATE,
    )

    model = create_mnist_model(max_epochs=EPOCHS, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = float(accuracy_score(y_train, train_pred))
    test_acc = float(accuracy_score(y_test, test_pred))

    model_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(model, model_path)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "epochs": int(EPOCHS),
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Результаты обучения:")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train() 