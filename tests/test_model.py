import os

from sklearn.metrics import accuracy_score

from src.data_loader import get_datasets, ensure_directory_exists
from src.model import create_mnist_model


def test_pipeline_trains_and_predicts(tmp_path):
    data_dir = os.path.join(tmp_path, "data")
    ensure_directory_exists(data_dir)

    x_train, y_train, x_test, y_test = get_datasets(
        dataset_directory=str(data_dir), train_subset=1000, test_subset=300, random_state=0
    )

    model = create_mnist_model(max_epochs=1, random_state=0)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    assert preds.shape[0] == x_test.shape[0]
    assert 0.0 <= acc <= 1.0 