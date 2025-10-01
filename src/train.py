import argparse
import json
import os
from typing import Dict

import joblib
from sklearn.metrics import accuracy_score

from src.data_loader import get_datasets, ensure_directory_exists
from src.model import create_mnist_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST model (lightweight)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--save-dir", type=str, default="./artifacts", help="Artifacts directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (passes over data)")
    parser.add_argument("--train-subset", type=int, default=5000, help="Number of train samples to use")
    parser.add_argument("--test-subset", type=int, default=1000, help="Number of test samples to use for quick eval")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train() -> Dict[str, float]:
    args = parse_args()

    ensure_directory_exists(args.save_dir)

    x_train, y_train, x_test, y_test = get_datasets(
        dataset_directory=args.data_dir,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        random_state=args.random_state,
    )

    model = create_mnist_model(max_epochs=args.epochs, random_state=args.random_state)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = float(accuracy_score(y_train, train_pred))
    test_acc = float(accuracy_score(y_test, test_pred))

    model_path = os.path.join(args.save_dir, "model.joblib")
    joblib.dump(model, model_path)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "epochs": int(args.epochs),
    }

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train() 