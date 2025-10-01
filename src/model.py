from typing import Optional

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_mnist_model(max_epochs: int = 1, random_state: int = 42) -> Pipeline:
    """Create a lightweight MNIST classifier pipeline suitable for CI.

    The pipeline standardizes features and trains a logistic regression-like
    linear classifier via SGD, which is fast and memory-efficient.
    """
    if max_epochs < 1:
        max_epochs = 1

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=max_epochs,
                    tol=1e-3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline 