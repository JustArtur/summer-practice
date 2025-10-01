import argparse
import json

import joblib
from sklearn.metrics import accuracy_score

from src.data_loader import get_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MNIST model")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model.joblib")
    parser.add_argument("--test-subset", type=int, default=1000, help="Number of test samples to use")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate() -> float:
    args = parse_args()
    _, _, x_test, y_test = get_datasets(
        dataset_directory=args.data_dir,
        train_subset=None,
        test_subset=args.test_subset,
        random_state=args.random_state,
    )

    model = joblib.load(args.model_path)
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))

    result = {"test_accuracy": acc, "test_samples": int(x_test.shape[0])}
    print(json.dumps(result, indent=2))
    return acc


if __name__ == "__main__":
    evaluate() 