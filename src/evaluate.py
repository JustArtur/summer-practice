import json
import os

import joblib
from sklearn.metrics import accuracy_score

from src.data_loader import get_datasets

DATA_DIR = "./data"
MODELS_DIR = "./models"
MODEL_FILENAME = "model.joblib"
TEST_SUBSET = 200  # Маленький датасет для быстрого прогона CI
RANDOM_STATE = 42


def evaluate() -> float:
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    print(f"Оцениваем модель: {model_path}")

    # Проверяем наличие модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    _, _, x_test, y_test = get_datasets(
        dataset_directory=DATA_DIR,
        train_subset=None,
        test_subset=TEST_SUBSET,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(model_path)
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))

    result = {
        "test_accuracy": acc,
        "test_samples": int(x_test.shape[0]),
        "model_path": model_path
    }

    print("Результаты оценки:")
    print(json.dumps(result, indent=2))
    return acc


if __name__ == "__main__":
    evaluate()
